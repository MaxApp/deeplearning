import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from pathlib import Path

import utils
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    current_dir = Path(__file__).parent

    # detect and show model classes support
    seg_model_weights = tv_models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    class_names, num_of_classes = utils.show_classes_from_weight_meta(seg_model_weights)
    print(f"Supported classes:\n {class_names}\n")
    print(f"Total: {num_of_classes}")

    if class_names is None:
        sys.exit()

    # initial model
    seg_model = tv_models.segmentation.deeplabv3_resnet50(weights=seg_model_weights).eval()

    image_path = str(current_dir / "imgs/horse1.png")
    img = Image.open(image_path)

    # Create the clean, un-normalized tensor for visualization later
    original_image_tensor = transforms.ToTensor()(img)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
    
    input_tensor = normalize(original_image_tensor).unsqueeze(0)

    # target classes
    target_class_names = ['horse']

    class_indices = [class_names.index(name) for name in target_class_names]

    # Print the results for confirmation
    print(f"Target Classes:        {target_class_names}")
    print(f"Corresponding Indices: {class_indices}")

    # prediction
    with torch.no_grad():
        output = seg_model(input_tensor)['out'][0]
    
    output_predictions = output.argmax(0)

    # Create a boolean mask
    individual_masks = [(output_predictions == i) for i in class_indices]
    stacked_masks = torch.stack(individual_masks, dim=0)

    # overlay generated masks onto the original image
    result = vutils.draw_segmentation_masks(image=(original_image_tensor * 255).byte(),
                                            masks=stacked_masks,
                                            alpha=0.5,
                                            colors=["green"])

    # show masked image with origin side by side
    fig, axes = plt.subplots(nrows=1,ncols=2)
    axes[0].imshow(img)
    axes[0].axis('off')
    # permute dimensions from (C, H, W) to (H, W, C)
    axes[1].imshow(result.permute(1, 2, 0))
    axes[1].axis('off')
    # Show the plot
    plt.tight_layout()
    plt.show()