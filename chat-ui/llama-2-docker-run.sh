# model=meta-llama/Llama-2-7b-chat-hf
model=meta-llama/Llama-2-13b-chat-hf
token="YOUR_HUGGINGFACE_ACCESS_TOKEN_HERE"
volume=$PWD/data

docker run --gpus all \
--shm-size 1g \
-e HUGGING_FACE_HUB_TOKEN=$token \
-p 8080:80 \
-v $volume:/data \
ghcr.io/huggingface/text-generation-inference:latest \
--model-id $model \
--quantize bitsandbytes-nf4 \
--max-input-length 2048 \
--max-total-tokens 4096
