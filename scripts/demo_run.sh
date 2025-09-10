#/bin/sh

BATCH_SIZE="${FUZZING_BATCH_SIZE:-1}"
MODEL_NAME="${FUZZING_MODEL:-"ollama/starcoder"}"
DEVICE="${FUZZING_DEVICE:-"gpu"}"

echo "BATCH_SIZE: $BATCH_SIZE"
echo "MODEL_NAME: $MODEL_NAME"
echo "DEVICE: $DEVICE"

if [ "$DEVICE" = "gpu" ]; then
    python Fuzz4All/fuzz.py --config config/cpp_demo.yaml main_with_config \
                            --folder outputs/demo/ \
                            --batch_size $BATCH_SIZE \
                            --model_name $MODEL_NAME \
                            --target /usr/bin/g++

elif [ "$DEVICE" = "vllm" ]; then
    python Fuzz4All/fuzz.py --config config/cpp_demo_vllm.yaml main_with_config \
                            --folder outputs/demo/ \
                            --batch_size $BATCH_SIZE \
                            --model_name $MODEL_NAME \
                            --target /usr/bin/g++ \
                            --use_vllm_server
else
    python Fuzz4All/fuzz.py --config config/cpp_demo.yaml main_with_config \
                            --folder outputs/demo/ \
                            --batch_size $BATCH_SIZE \
                            --model_name $MODEL_NAME \
                            --cpu \
                            --target /usr/bin/g++
fi
