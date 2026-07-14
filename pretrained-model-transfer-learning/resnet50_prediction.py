import torch
import torchvision
import torchvision.transforms.v2 as transforms
from pathlib import Path
import utils
import sys
import matplotlib.pyplot as plt

if "__main__" == __name__:
    current_dir = Path(__file__).parent
    # load class names with json file
    class_names_dict = utils.load_classnames_json(str(current_dir / "./imagenet_class_index.json"))

    # load resnet50 model with pre-downloaded .pth
    # you need to set path to your own，I do not upload the .pth file because of the large size
    pretrained_param_file_path = "./resnet50-0676ba61.pth"  
    resnet50 = torchvision.models.resnet50(weights=None)
    try:
        state_dict = torch.load(pretrained_param_file_path, map_location="cpu")
    except Exception as e:
        print(f"Loading pre-trained parameters file failed: {e}")
        sys.exit()

    resnet50.load_state_dict(state_dict)
    resnet50.eval()

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    image_files = ["imgs/img1.jpg","imgs/img2.jpg","imgs/img3.jpg","imgs/img4.png"]
    with torch.no_grad():
        trans = transforms.Compose(
                [
                    # if use `decode_image`, the value of pixels is uint8
                    # we need to convert it to float32 and normalized
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Resize(232),
                    transforms.CenterCrop(224),
                    transforms.Normalize(mean, std)
                ]

            )
        
        # make predictions with pretrained model
        original_image = []
        image_names = []
        for i in range(len(image_files)):
            image = torchvision.io.decode_image(str(current_dir / image_files[i]))
            img_tensor = trans(image)
            original_image.append(img_tensor)

            predict = resnet50(img_tensor.unsqueeze(0))
            p = torch.nn.functional.softmax(predict[0], dim=0)

            top_prob, top_catid = torch.topk(p, 1)
            catid_or_name = str(top_catid.item())
            if class_names_dict is not None:
                catid_or_name = class_names_dict[catid_or_name][1]
            image_names.append(catid_or_name)
            print(f"File: {image_files[i]}, ClassName: {catid_or_name}, Probability: {top_prob.item()}")

    # show the images with names
    fig, axes = plt.subplots(nrows=1, ncols=4)
    for i, img in enumerate(original_image):
        # denormalize cropped image tensor to show up 
        img = img * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)
        axes[i].imshow(img.permute(1,2,0))
        axes[i].set_title(image_names[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()