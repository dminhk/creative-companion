#############################################################################
# imports
#############################################################################

import os
 
import numpy as np
import torch
import torch.nn as nn
import torch_neuronx
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
 
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import time
import copy
from IPython.display import clear_output

clear_output(wait=False)

#############################################################################
# Define utility classes and functions
#############################################################################

def get_attention_scores_neuron(self, query, key, attn_mask):    
    if(query.size() == key.size()):
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2),
            self.scale
        )
        attention_probs = attention_scores.softmax(dim=1).permute(0,2,1)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2),
            self.scale
        )
        attention_probs = attention_scores.softmax(dim=-1)
  
    return attention_probs
 

def custom_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled
 

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
 
    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        out_tuple = self.unet(sample,
                              timestep,
                              encoder_hidden_states,
                              added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                              return_dict=False)
        return out_tuple
    
    
class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device
 
    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None):
        sample = self.unetwrap(sample,
                               timestep.float().expand((sample.shape[0],)),
                               encoder_hidden_states,
                               added_cond_kwargs["text_embeds"],
                               added_cond_kwargs["time_ids"])[0]
        return UNet2DConditionOutput(sample=sample)

#############################################################################
# Compile the model into an optimized TorchScript and save the TorchScript
#############################################################################

COMPILER_WORKDIR_ROOT = 'sdxl_compile_dir_1024'

# Model ID for SD XL version pipeline
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# --- Compile VAE decoder and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
decoder = copy.deepcopy(pipe.vae.decoder)
del pipe

# # Compile vae decoder
decoder_in = torch.randn([1, 4, 128, 128])
decoder_neuron = torch_neuronx.trace(
    decoder, 
    decoder_in, 
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
)

# Save the compiled vae decoder
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
torch.jit.save(decoder_neuron, decoder_filename)

# delete unused objects
del decoder


# --- Compile UNet and save ---

print("Load model....")
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# --- Load LoRA model HERE ---
print("Load LoRA model....")
pipe.load_lora_weights("dminhk/dog-example-sdxl-lora")

# Replace original cross-attention module with custom cross-attention module for better performance
Attention.get_attention_scores = get_attention_scores_neuron

# Apply double wrapper to deal with custom return type
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))

# Only keep the model being compiled in RAM to minimze memory pressure
unet = copy.deepcopy(pipe.unet.unetwrap)
del pipe

# Compile unet - FP32
sample_1b = torch.randn([1, 4, 128, 128])
timestep_1b = torch.tensor(999).float().expand((1,))
encoder_hidden_states_1b = torch.randn([1, 77, 2048])
added_cond_kwargs_1b = {"text_embeds": torch.randn([1, 1280]),
                        "time_ids": torch.randn([1, 6])}
example_inputs = (sample_1b, timestep_1b, encoder_hidden_states_1b, added_cond_kwargs_1b["text_embeds"], added_cond_kwargs_1b["time_ids"],)

unet_neuron = torch_neuronx.trace(
    unet,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
    compiler_args=["--model-type=unet-inference"]
)

# save compiled unet
unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
torch.jit.save(unet_neuron, unet_filename)

# delete unused objects
del unet


# --- Compile VAE post_quant_conv and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
del pipe

# Compile vae post_quant_conv
post_quant_conv_in = torch.randn([1, 4, 128, 128])
post_quant_conv_neuron = torch_neuronx.trace(
    post_quant_conv, 
    post_quant_conv_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
)

# Save the compiled vae post_quant_conv
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)

# delete unused objects
del post_quant_conv

