# lora-training-ui.py

import streamlit as st
import os
import subprocess
st.title("Stable Diffusion XL LoRA fine-tuning")
uploaded_files = st.file_uploader("Choose training images", accept_multiple_files=True)
if uploaded_files:
    # Save uploaded files 
    os.makedirs("training_images", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("training_images", file.name),"wb") as f:
            f.write(file.getbuffer())
    st.write("Training images saved!")
    # Allow user to customize instance prompt
    instance_prompt = st.text_input("Instance prompt", "a photo of sks dog")
    # Build training command
    train_cmd = f"""
    rm -r training_images/.ipynb_checkpoints
    export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
    export INSTANCE_DIR="training_images"
    export OUTPUT_DIR="lora-trained-xl"
    accelerate launch train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="fp16" \
    --instance_prompt="{instance_prompt}" \
    --resolution=1024 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --enable_xformers_memory_efficient_attention \
    --use_8bit_adam \
    --max_train_steps=500 \
    --seed="0" 
    """
    # Run training and display logs
if st.button("Start Training"):
    st.write("Running training command...")
    #process = subprocess.Popen(train_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    process = subprocess.Popen(train_cmd, shell=True, stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, b""):
      st.write(line.decode('utf-8'))
    st.write("Done!")
