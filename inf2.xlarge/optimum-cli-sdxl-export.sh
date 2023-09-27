optimum-cli export neuron --model stabilityai/stable-diffusion-xl-base-1.0 \
--task stable-diffusion-xl \
--batch_size 1 \
--height 1024 \
--width 1024 \
--auto_cast matmul \
--auto_cast_type bf16 \
stable-diffusion-xl-base-1.0-neuronx/
