#!/usr/bin/env bash
# Start SGLang HTTP server on Tenstorrent (TT) backend
# Mirrors vLLM plugin serve.sh defaults where applicable.

set -euo pipefail

: "${HF_MODEL:=meta-llama/Llama-3.1-8B-Instruct}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"

# Optional: choose mesh grid like vLLM (e.g., N150, N300, "(1, 2)")
# export MESH_DEVICE=${MESH_DEVICE:-N150}

# Ensure local sources are importable if running from repo
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/sglang/python:$(pwd)/tt-metal

exec python3 -m sglang.launch_server \
  --model-path "${HF_MODEL}" \
  --device tt \
  --host "${HOST}" \
  --port "${PORT}"
