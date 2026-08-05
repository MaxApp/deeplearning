import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_image(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def compute_saliency_map(model, input_image, target_class=None):
    """
    computes a saliency map for an input image
    """
    # enable gradient tracking for input
    input_image = input_image.clone().detach()
    input_image.requires_grad_()

    output = model(input_image)

    # calculate probabilities
    probs = torch.softmax(output, dim=1)
    if target_class is None:
        pred_prob, pred_class = torch.max(probs, dim=1)
        target_class = pred_class.item()
        pred_prob = pred_prob.item()
    else:
        pred_prob = probs[0, target_class].item()
        pred_class = target_class

    model.zero_grad()

    output[0, target_class].backward()
    gradients = input_image.grad[0]

    # sum up channels for a 2D map
    saliency_map = torch.abs(gradients).sum(dim=0)

    # standardization
    saliency_map = (saliency_map - saliency_map.min()) / (
        saliency_map.max() - saliency_map.min() + 1e-8)

    return saliency_map, pred_class, pred_prob


def visualize_saliency(img_pil, saliency_map):
    """
    Displays the original image along with an enhanced saliency map
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # the original image
    ax1.imshow(img_pil)
    ax1.set_title(f'Original Image')
    ax1.axis('off')

    # enhance the contrast of the saliency map
    gamma = 0.7
    saliency_map_enhanced = torch.pow(saliency_map, gamma)

    # resize saliency map to match the source image
    w, h = img_pil.size
    # saliency_map_resized = resize(
    #     saliency_map_enhanced, (h, w),
    #     order=1, mode='reflect', anti_aliasing=True
    # )

    saliency_map_resized = F.interpolate(
        saliency_map_enhanced.unsqueeze(0).unsqueeze(0), 
        size=(h, w), 
        mode='bilinear',
        align_corners=False,
        antialias=True
    )

    # Plot the enhanced saliency heatmap in the second panel
    saliency_heatmap = ax2.imshow(saliency_map_resized[0].permute(1,2,0), cmap='inferno')
    ax2.set_title('Enhanced Saliency Map')
    ax2.axis('off')
    # fig.colorbar(saliency_heatmap, ax=ax2, fraction=0.046)

    plt.tight_layout()
    plt.show()


if "__main__" == __name__:
    current_dir = Path(__file__).parent
    pretrained_param_file_path = "E:\\PDF\\pytorch\\resnet50-0676ba61.pth" #"./resnet50-0676ba61.pth"  
    resnet50 = tv_models.resnet50(weights=None)
    try:
        state_dict = torch.load(pretrained_param_file_path, map_location="cpu")
    except Exception as e:
        print(f"Loading pre-trained parameters file failed: {e}")
        sys.exit()
    resnet50.load_state_dict(state_dict)
    # set model to evaluate mode
    resnet50.eval()

    # class names
    imagenet_class_mapping = tv_models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

    # transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    img_path = current_dir / 'imgs/cat.jpg'

    if not Path(img_path).exists():
        raise FileNotFoundError("Image not found")

    img_pil = load_image(img_path)
    img_tensor = transform(img_pil).unsqueeze(0)

    saliency_map, pred_class, pred_prob = compute_saliency_map(resnet50, img_tensor)
    visualize_saliency(img_pil, saliency_map)