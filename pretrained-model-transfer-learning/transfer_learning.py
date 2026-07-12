import torch
import torchvision
import torchvision.transforms.v2 as transforms
from PIL import Image

# load resnet50 model with pre-downloaded .pth 
resnet50 = torchvision.models.resnet50(weights=None)
state_dict = torch.load("E:\\PDF\\pytorch\\resnet50-0676ba61.pth", map_location="cpu")
resnet50.load_state_dict(state_dict)
resnet50.eval()
with torch.no_grad():
    image_files = ["E:\\img1.jpg","E:\\img2.jpg","E:\\img3.png"]
    trans = transforms.Compose(
            [
                # if use `decode_image`, the value of pixels is uint8
                # we need to convert it to float32 and normalized
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize(232),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]

        )
    
    # make predictions with pretrained model
    for i in range(3):
        image = torchvision.io.decode_image(image_files[i])
        img_tensor = trans(image)
        
        predict = resnet50(img_tensor.unsqueeze(0))
        p = torch.nn.functional.softmax(predict[0], dim=0)

        top_prob, top_catid = torch.topk(p, 1)
        print(f"file:{image_files[i]}, p: {top_prob}  c:{top_catid}")

