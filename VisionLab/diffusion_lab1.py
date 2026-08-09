from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
import torch
import os

if __name__ == "__main__":
    """Generate 3 images by prompt words and display with matplotlib"""

    # model id and cache directory
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    cache_dir = os.path.join(os.getcwd(), './models')
    model_cache_dir_name = f"models--{repo_id.replace('/', '--')}"
    model_cache_path = os.path.join(cache_dir, model_cache_dir_name)
    model_is_cached = os.path.isdir(model_cache_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = DiffusionPipeline.from_pretrained(
        repo_id, 
        torch_dtype=torch.float16, 
        variant="fp16", 
        cache_dir=cache_dir,
        local_files_only=model_is_cached
    )

    # set the seed
    generator = torch.Generator(device=device).manual_seed(42)
    pipe = pipe.to(device)

    # generate image by prompt
    prompt="Astronaut in a jungle, cold color palette, muted colors, detailed"

    images = pipe(
        prompt=prompt,
        num_inference_steps=20,  # steps
        generator=generator,
        width=768,
        height=768,
        num_images_per_prompt = 3
    ).images

    # display images
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    for idx, img in enumerate(images):
        axs[idx].imshow(img)
        axs[idx].axis('off')

    plt.tight_layout()
    plt.show()