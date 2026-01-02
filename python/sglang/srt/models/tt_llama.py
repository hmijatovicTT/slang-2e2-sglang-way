import torch
from torch import nn
import ttnn
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from models.tt_transformers.tt.generator_sglang import LlamaForCausalLM as TT_Llama
from models.tt_transformers.tt.model_config import DecodersPrecision
from sglang.srt.utils.tt_utils import open_mesh_device
import logging

logger = logging.getLogger(__name__)

class TTLlamaForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, tt_model=None, **kwargs):
        super().__init__()
        self.config = config
        self.kv_caches = None
        self.block_size = 32 # TT standard

        if tt_model is not None:
            self.tt_model = tt_model
        else:
            # Initialize TT model if not provided
            # Default parameters
            max_batch_size = 32
            
            # Determine max_seq_len from config
            if hasattr(config, "max_sequence_length"):
                max_seq_len = config.max_sequence_length
            elif hasattr(config, "max_position_embeddings"):
                max_seq_len = config.max_position_embeddings
            else:
                max_seq_len = 4096 # Fallback
                
            n_layers = None
            tt_data_parallel = 1
            optimizations = "performance"
            override_tt_config = None

            # Initialize TT device
            try:
                if torch.distributed.is_initialized():
                    rank = torch.distributed.get_rank()
                else:
                    rank = 0
            except Exception:
                rank = 0

            mesh_device = open_mesh_device(override_tt_config, trace_mode=False, dp_rank=rank)
            
            self.tt_model = TT_Llama.initialize_vllm_model(
                config,
                mesh_device,
                max_batch_size,
                max_seq_len,
                n_layers,
                tt_data_parallel,
                optimizations
            )

        logger.info(
            f"TT_Llama.initialize_vllm_model executed"
        )

    @classmethod
    def initialize_sglang_model(
        cls,
        hf_config,
        device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations="performance",
        override_tt_config=None,
    ):
        # Initialize TT device
        try:
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
        except Exception:
            rank = 0

        mesh_device = open_mesh_device(override_tt_config, trace_mode=False, dp_rank=rank)
        
        # Note: We are not closing the device here as it needs to stay open for the model.
        
        tt_model = TT_Llama.initialize_vllm_model(
            hf_config,
            mesh_device,
            max_batch_size,
            max_seq_len,
            n_layers,
            tt_data_parallel,
            optimizations
        )
        return cls(config=hf_config, tt_model=tt_model)

    def load_weights(self, weights):
        # TT model loads weights during initialization
        pass

    def allocate_kv_cache(self, kv_cache_shape, dtype=None, layer_num=None):
        if self.kv_caches is None:
            # We need to extract block size from the shape if possible, or use default
            # Shape is usually (num_blocks, num_kv_heads, block_size, head_dim)
            if len(kv_cache_shape) >= 3:
                self.block_size = kv_cache_shape[2]
            
            # Use layer_num if provided (SGLang passes total layers), otherwise get from model args
            num_layers = layer_num if layer_num is not None else self.tt_model.model_args[0].n_layers
            
            # Delegate to TT model's allocate_kv_cache (same as vLLM plugin)
            self.kv_caches = self.tt_model.allocate_kv_cache(
                kv_cache_shape=kv_cache_shape,
                dtype=dtype,
                num_layers=num_layers
            )
        return self.kv_caches

    def _build_page_table(self, forward_batch):
        req_to_token_pool = forward_batch.req_to_token_pool
        req_pool_indices = forward_batch.req_pool_indices
        
        # (batch_size, max_len)
        batch_req_tokens = req_to_token_pool.req_to_token[req_pool_indices]
        
        # Subsample to get block indices
        # We take the first token index of each block.
        # (batch_size, max_blocks)
        page_table = batch_req_tokens[:, ::self.block_size] // self.block_size
        return page_table

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> LogitsProcessorOutput:
        
        page_table = self._build_page_table(forward_batch)
        
        if forward_batch.forward_mode.is_extend():
            # Prefill
            # sglang input_ids is flattened (total_tokens,)
            # We need to reconstruct batch structure (batch, max_len)
            
            batch_size = forward_batch.batch_size
            seq_lens = forward_batch.seq_lens
            max_len = torch.max(seq_lens).item()
            
            # Create padded tokens tensor
            padded_tokens = torch.zeros((batch_size, max_len), dtype=torch.long, device=input_ids.device)
            
            start = 0
            for i in range(batch_size):
                length = seq_lens[i].item()
                padded_tokens[i, :length] = input_ids[start:start+length]
                start += length
            
            # TT expects tokens on host usually, but can handle device tensors if mapped?
            # generator.py usually moves to device.
            # We pass what we have.
            
            logits = self.tt_model.prefill_forward(
                tokens=padded_tokens,
                page_table=page_table,
                kv_cache=self.kv_caches,
                prompt_lens=seq_lens,
                enable_trace=False 
            )

            logger.info(
            f"self.tt_model.prefill_forward executed"
            )
            
            # logits is (batch, 1, vocab_size)
            return LogitsProcessorOutput(next_token_logits=logits.squeeze(1))

        elif forward_batch.forward_mode.is_decode():
            # Decode
            # input_ids is (batch,)
            tokens = input_ids.unsqueeze(1)
            start_pos = positions  # (batch,)

            # TT decode expects per-device batch == max_batch_size; pad then slice back
            dp = getattr(self.tt_model, "data_parallel", len(self.tt_model.model))
            max_bsz = self.tt_model.model_args[0].max_batch_size
            required_bsz = dp * max_bsz
            actual_bsz = tokens.shape[0]
            if actual_bsz > required_bsz:
                raise ValueError(f"Decode batch {actual_bsz} exceeds TT capacity {required_bsz}")

            if actual_bsz < required_bsz:
                pad_n = required_bsz - actual_bsz
                # Pad tokens (long), positions (long), and page_table (int32) with zeros
                pad_tokens = torch.zeros((pad_n, 1), dtype=tokens.dtype, device=tokens.device)
                tokens = torch.cat([tokens, pad_tokens], dim=0)

                pad_pos = torch.zeros((pad_n,), dtype=start_pos.dtype, device=start_pos.device)
                start_pos = torch.cat([start_pos, pad_pos], dim=0)

                if page_table is not None:
                    pt_width = page_table.shape[1]
                    pad_pt = torch.zeros((pad_n, pt_width), dtype=page_table.dtype, device=page_table.device)
                    page_table = torch.cat([page_table, pad_pt], dim=0)

            logits = self.tt_model.decode_forward(
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table,
                kv_cache=self.kv_caches,
                enable_trace=False,
                read_from_device=True,
            )

            logger.info(
            f"self.tt_model.decode_forward executed"
            )

            # Slice to actual batch, then squeeze sequence dim
            logits = logits[:actual_bsz]
            return LogitsProcessorOutput(next_token_logits=logits.squeeze(1))
            
            
        else:
            raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")

EntryClass = [TTLlamaForCausalLM]
