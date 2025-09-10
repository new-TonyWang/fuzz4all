export VLLM_USE_V1=1
vllm serve \
    /home/zinc/models/starcoderbase-1b \
    --served-model-name starcoderbase-1b \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.8 \
    --enable-prefix-caching &> vllm_server.log &