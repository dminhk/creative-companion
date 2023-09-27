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
# Load the saved model and run it
#############################################################################

# --- Load all compiled models ---
COMPILER_WORKDIR_ROOT = 'sdxl_compile_dir_1024'
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Load the compiled UNet onto two neuron cores.
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0,1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

# Load other compiled models onto a single neuron core.
pipe.vae.decoder = torch.jit.load(decoder_filename)
pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)

# Run pipeline
prompt = ["Magazine Cover, a sks dog in a candid pose wearing funky clothes, half body photoshoot, Magazine photoshoot, Studio Lighting, Creative trippy aesthetic background, Modern look",]

# First do a warmup run so all the asynchronous loads can finish
image_warmup = pipe(prompt[0]).images[0]

plt.title("Image")
plt.xlabel("X pixel scaling")
plt.ylabel("Y pixels scaling")

total_time = 0
for x in prompt:
    start_time = time.time()
    image = pipe(x).images[0]
    total_time = total_time + (time.time()-start_time)
    image.save("image.png")
#    image = mpimg.imread("image.png")
    #clear_output(wait=True)
#    plt.imshow(image)
#    plt.show()
print("Average time: ", np.round((total_time/len(prompt)), 2), "seconds")
