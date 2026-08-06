import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as tv_models
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pathlib import Path
import numpy as np
import sys

import utils

class GradCAM:

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        # storage for activation maps
        self.activations = None   # [N,C,H',W']
        # storage for gradients
        self.gradients = None     # [N,C,H',W']
        # register the forward hook
        self.target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self.activations = output.detach()
        # register a hook on the output
        output.register_hook(self._on_backward)

    def _on_backward(self, grad):
        self.gradients = grad.detach()

    def __call__(self, image_batch: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad(set_to_none=True)
        output = self.model(image_batch)  # logits [1, num_classes]
        # Determine the target class index if not provided
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        score = output[:, class_idx].sum()
        score.backward()

        # compute average pooling of the gradients
        # activations/gradients are [1, C, H', W']
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # [1,C,1,1]
        cam = (weights * self.activations).sum(dim=1, keepdim=False) # [1,H',W']
        # keep positive influence
        cam = cam.relu()[0]  

        # standarization
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        
        # the final heatmap
        return cam.detach().numpy()


def compute_gradcam(img_tensor, model):
    try:
        # forward pass
        output = model(img_tensor)
        pred_class_idx = torch.argmax(output, dim=1).item()
        # pred_score = torch.softmax(output, dim=1)[0, pred_class_idx].item()
        # Map class index to human-readable label
        # pred_class_name = imagenet_class_mapping[pred_class_idx]

        # GradCAM calculation by resnet50
        grad_cam = GradCAM(model, model.layer4[-1].conv3)
        # Generate GradCAM heatmap for the predicted class
        heatmap = grad_cam(img_tensor, int(pred_class_idx))  # (activation_h, activation_w)
        return heatmap #, pred_class_name, pred_score

    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None


def visualize_gradcam(img_display, heatmap):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # 1: Original image
    ax1.imshow(img_display)
    ax1.set_title(f'Original')
    ax1.axis('off')

    # 2: Standalone heatmap
    ax2.imshow(heatmap, cmap='jet')
    ax2.set_title('GradCAM')
    ax2.axis('off')

    # make heatmap overlay
    img_display_np = np.array(img_display)
    heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)
    heatmap_resized = TF.resize(
        heatmap_tensor,
        size=(img_display_np.shape[0], img_display_np.shape[1]),                       # size 参数为 (H, W)
        interpolation=TF.InterpolationMode.BILINEAR
    ).squeeze(0).numpy()

    jet_cmap = colormaps['jet']
    heatmap_norm = np.clip(heatmap_resized, 0, 1)
    heatmap_color = jet_cmap(heatmap_norm)[:, :, :3]   # (H, W, 3) RGB
    heatmap_color = (heatmap_color * 255).astype(np.uint8)

    superimposed = (
        img_display_np.astype(np.float32) * 0.6
        + heatmap_color.astype(np.float32) * 0.4
    ).astype(np.uint8)
    
    ax3.imshow(superimposed)
    ax3.set_title("Overlay")
    ax3.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    pretrained_param_file_path = "E:\\PDF\\pytorch\\resnet50-0676ba61.pth" # "./resnet50-0676ba61.pth"  # change to your own path
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
    
    # load the image
    img = utils.load_image(img_path)
    img_tensor = transform(img).unsqueeze(0)

    heat_map = compute_gradcam(img_tensor, resnet50)

    visualize_gradcam(img.resize((224,224)), heat_map)