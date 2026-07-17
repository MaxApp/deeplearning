import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as tv_models
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
import random
import utils


# MNIST mean and std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def show_menu():
    print()
    print("=== Choose numbers below to run ===")
    print("1. directly make prediction without any change")
    print("2. replace classifier and post-train, then make prediction")
    print("0. exit")
    print()


def predict_digitals(model, dataset, class_name_dict):
    random.seed(42)
    total = len(dataset)
    # get random 10 indicies from dataset to make prediction
    sample_size = 10
    indices = [random.randint(0, total-1) for _ in range(sample_size)]
    fig = plt.figure()

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, true_label = dataset[idx]
            output = model(img.unsqueeze(0))
            pred = torch.argmax(output).item()
            if class_name_dict:
                pred = class_name_dict[str(pred)][1]
            print(f"true label: {true_label}  pred: {pred}")

            # plot images
            axe = fig.add_subplot(2,5,i+1)
            axe.axis('off')
            axe.set_title(str(pred))

            mean = torch.tensor(MEAN).view(3, 1, 1)
            std = torch.tensor(STD).view(3, 1, 1)
            axe.imshow((img * std + mean).permute(1,2,0))
    plt.tight_layout()
    plt.show()


def load_model_with_weights():
    # load model with pretrained weights
    return tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)

def mnist_dataloader() -> tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader]:
    # load MNIST train data and wrapped with dataloader
    mnist_transform = transforms.Compose([
        # Convert grayscale image to 3 channels to match MobileNetV2's input
        transforms.Grayscale(num_output_channels=3),
        # Resize the image to 224x224, the standard input size for MobileNetV2
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize the tensor using ImageNet's mean and standard deviation
        transforms.Normalize(mean=MEAN, std=STD)
        # transforms.Normalize(mean=MEAN, std=STD)  
    ])
    train_dataset = torchvision.datasets.MNIST(root="./data",train=True,download=False,transform=mnist_transform)
    return train_dataset, DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


if __name__ == "__main__":

    show_menu()
    choose = input("Choose: ").strip()

    if choose == '1': # directly make predictions without any change
        # load ImageNet class name indecies
        current_dir = Path(__file__).parent
        class_names_dict = utils.load_classnames_json(str(current_dir / "./imagenet_class_index.json"))

        # load model and data
        model = load_model_with_weights()
        dataset, dataloader = mnist_dataloader()

        print("== Prediction ==")
        predict_digitals(model, dataset, class_names_dict)

    elif choose == '2': # replace the classifier, post-trained and make predictions
        # load model
        model = load_model_with_weights()

        print("")
    else:
        print("terminated")

    
    
    

    # Case 1: Directly make predictions on MNIST



    # Case 2: Substitude the classifier layer for 10 item classifications 
    #         and do post-training and fine tuning, then make predictions

"""
    # freeze pretrained parameters for feature layers
    for feature_parameter in mobilenet_model.features.parameters():
        feature_parameter.requires_grad = False

    # the final classification layer
    last_classifier_layer = mobilenet_model.classifier[-1]
    # the in_features of last_classifier_layer
    num_features = last_classifier_layer.in_features

    # our target classifications for MINIST
    num_classes = 26

    # substitude for a new layer
    mobilenet_model.classifier[-1] = nn.Linear(in_features=num_features, out_features=num_classes)

    # training strategy
    loss_function = torch.nn.CrossEntropyLoss()
    # Only optimize the parameters that require gradients
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, mobilenet_model.parameters()), lr=0.001)
"""