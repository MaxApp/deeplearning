import torch
import torchvision.io
import torchvision.models as tv_models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from pathlib import Path
import utils
import sys
import matplotlib.pyplot as plt


if "__main__" == __name__:
    current_dir = Path(__file__).parent
    image_path = str(current_dir / 'imgs/street_cars_01.png')

    # instantiate model with pre-trained weights
    weights = tv_models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    detection_model = tv_models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights).eval()

    # get class names from meta
    class_names, num_classes = utils.show_classes_from_weight_meta(weights_obj=weights)
    if class_names is None:
        sys.exit()

    # target classes
    target_class_names = ['car']

    # colors for bounding box
    box_colors = ['red']

    # target indices
    object_indices = [class_names.index(name) for name in target_class_names]

    image_tensor = torchvision.io.decode_image(image_path)
    # NO, you can't do normalize, the model's input only need scaled tensor
    # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    image_tensor = image_tensor / 255.0

    # Perform inference to get predictions for all possible objects
    with torch.no_grad():
        # prediction contains:
        # - boxes
        # - labels
        # - scores
        prediction = detection_model(image_tensor.unsqueeze(0))[0]

    # Initialize lists to collect all boxes, labels, and colors that meet the criteria
    all_boxes_to_draw = []
    all_labels_to_draw = []
    all_colors_to_draw = []

    # set a threshold
    threshold = 0.7

    for index, label, color in zip(object_indices, target_class_names, box_colors):
        # Filter predictions with confidence threshold
        class_mask = (prediction['labels'] == index) & (prediction['scores'] > threshold)
        boxes_for_this_class = prediction['boxes'][class_mask]

        if boxes_for_this_class.nelement() > 0:
            # Add the found boxes
            all_boxes_to_draw.extend(boxes_for_this_class.tolist())
            all_labels_to_draw.extend([label] * len(boxes_for_this_class))
            all_colors_to_draw.extend([color] * len(boxes_for_this_class))

    # After checking all classes, draw all collected boxes at once if any were found
    if all_boxes_to_draw:
        result_image_tensor = vutils.draw_bounding_boxes(
            image_tensor,
            torch.tensor(all_boxes_to_draw),
            labels=all_labels_to_draw,
            colors=all_colors_to_draw,
            width=3
        )

        # show the detection results with origin side by side
        fig, axes = plt.subplots(nrows=1,ncols=2)
        axes[0].imshow(image_tensor.permute(1,2,0))
        axes[0].axis('off')
        # permute dimensions from (C, H, W) to (H, W, C)
        axes[1].imshow(result_image_tensor.permute(1, 2, 0))
        axes[1].axis('off')
        # Show the plot
        plt.tight_layout()
        plt.show()
    else:
        print(f"No objects is detected.")