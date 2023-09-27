# sdxl-lora-inference-base-refiner.py
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch
# Load the base pipeline and load the LoRA parameters into it.
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16)
pipe.to("cuda")
pipe.load_lora_weights(
    "dog-example-xl-lora/", weight_name="pytorch_lora_weights.safetensors"
)
# pipe.load_lora_weights("dminhk/dog-example-xl-lora")

# Load the refiner.
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
     "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16)
refiner.to("cuda")
# SDXL styles: enhance, anime, photographic, digital-art, comic-book, fantasy-art, line-art, analog-film, neon-punk, isometric, low-poly, origami, modeling-compound, cinematic, 3d-mode, pixel-art, and tile-texture
prompt = "a photo of sks dog in a bucket"
for seed in range(4):
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(prompt=prompt, generator=generator, num_inference_steps=25)
    image = image.images[0]
    image = refiner(prompt=prompt, generator=generator, image=image)
    image = image.images[0]
    image.save(f"sdxl-base-refiner-{seed}.png")
