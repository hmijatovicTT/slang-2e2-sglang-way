#!/usr/bin/env bash
# Simple online benchmark against a running SGLang server on TT backend.
# This is a lightweight placeholder analogous to vLLM's online_benchmark.sh.

set -euo pipefail

: "${HOST:=localhost}"
: "${PORT:=8000}"
: "${NUM_PROMPTS:=50}"

URL="http://${HOST}:${PORT}/generate"

prompts=(
  "Tell me a joke."
  "What is the capital of Italy?"
  "Explain quantum computing in simple terms."
  "How do you make pancakes?"
  "What is the tallest mountain?"
)

count=0
for prompt in "${prompts[@]}"; do
  for i in $(seq 1 10); do
    ((count++))
    if (( count > NUM_PROMPTS )); then break; fi
    curl -sS -X POST "${URL}" \
      -H "Content-Type: application/json" \
      -d "{\"text\": \"${prompt}\", \"sampling_params\": {\"max_new_tokens\": 32, \"temperature\": 0.0}, \"stream\": false}" \
      > /dev/null
  done
  if (( count > NUM_PROMPTS )); then break; fi
  sleep 0.2
done

echo "Sent ${count} requests to ${URL}"