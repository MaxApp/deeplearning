# from diffusers import StableDiffusionXLPipeline
from diffusers import DiffusionPipeline, AutoencoderKL
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import numpy as np


def denoise_callback(step_index: int, timestep: int, latents: torch.Tensor):
    with torch.no_grad():
        # SDXL scaling factor：0.13025
        latents_input = latents / 0.13025
        image = pipe.vae.decode(latents_input).sample
        image = (image / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    pil_image = Image.fromarray((image * 255).astype("uint8"))
    all_steps.append((step_index, pil_image))

def plot_process():
    print(f"total steps: {len(all_steps)}")
    fig, axes = plt.subplots(1, 5)
    fig.suptitle('Denoising Steps', fontsize=14)

    for idx, (step, img) in enumerate(all_steps):
        axes[idx].imshow(img)
        axes[idx].set_title(f"Step {step}", fontsize=8)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    """Visualize process of denoising with callback functions"""
    all_steps = []

    # model id and cache directory
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    cache_dir = os.path.join(os.getcwd(), './models')
    model_cache_dir_name = f"models--{repo_id.replace('/', '--')}"
    model_cache_path = os.path.join(cache_dir, model_cache_dir_name)
    model_is_cached = os.path.isdir(model_cache_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fixed for float16 from community
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16
    )

    pipe = DiffusionPipeline.from_pretrained(
        repo_id, 
        vae=vae,  # substitude with AutoencoderKL
        torch_dtype=torch.float16, 
        variant="fp16", 
        cache_dir=cache_dir,
        local_files_only=model_is_cached
    )

    # set the seed
    generator = torch.Generator(device=device).manual_seed(42)
    pipe = pipe.to(device)

    # generate image by prompt
    prompt="a cute monkey with a yellow bandana, sitting in a lush park"

    images = pipe(
        prompt=prompt,
        num_inference_steps=21,  # steps
        generator=generator,
        width=768,
        height=768,
        callback=denoise_callback,
        callback_steps=5  # every five steps
    ).images

    # display process
    plot_process()