curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "starcoderbase-1b",
    "prompt": "Write a Python implementation of quicksort.",
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false,
    "n": 20
  }'