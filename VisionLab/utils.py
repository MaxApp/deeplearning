import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_image(image_tensor, title="", **kwargs):
    """
    Plots a tensor image using matplotlib.

    Args:
        image_tensor (torch.Tensor): A tensor representing an image with shape (C, H, W).
        title (str, optional): The title of the plot. Defaults to None.
        **kwargs: Additional keyword arguments passed to `plt.imshow`.
    """
    image_np = image_tensor.squeeze(0).numpy()
    img_transposed = np.transpose(image_np, (1, 2, 0))
    
    # Use the 'nearest' interpolation to enhance pixelation
    plt.imshow(img_transposed, interpolation='nearest', **kwargs)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_channels(out_tensor):
    # Loop over each output channel (filter)
    channels = out_tensor.shape[0]
    for i in range(channels):
        # Determine the grid size based on total number of filters
        grid_size = int(np.ceil(np.sqrt(channels)))
        # Add a subplot in a grid layout for each filter
        plt.subplot(grid_size, grid_size, i + 1)
        # Detach the tensor from the computation graph, convert to numpy array for visualization
        plt.imshow(out_tensor[i].detach().numpy(), cmap='gray')
        # Remove axis for a cleaner look
        plt.axis('off')
        # Set the title for each filter with proper formatting
        plt.title(f'Filter {i+1}', fontsize=10, pad=10)  

    # Adjust layout to prevent overlap
    plt.tight_layout()  
    # Display the plot with all filters
    plt.show()

def load_image(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img