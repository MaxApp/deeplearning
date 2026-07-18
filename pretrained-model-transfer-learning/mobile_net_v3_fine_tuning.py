import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as tv_models
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
import random
import utils


# ImageNet mean and std
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
            output = model(img.unsqueeze(0)) # remember to add batch dim
            pred = torch.argmax(output).item()
            if class_name_dict:
                pred = class_name_dict[str(pred)][1]
            print(f"true label: {true_label}  pred: {pred}")

            # plot images
            axe = fig.add_subplot(2,5,i+1)  # subplot index starts from 1
            axe.axis('off')
            axe.set_title(str(pred))

            # convert shape and de-normalize
            mean = torch.tensor(MEAN).view(3, 1, 1)
            std = torch.tensor(STD).view(3, 1, 1)
            axe.imshow((img * std + mean).permute(1,2,0))

    plt.tight_layout()
    plt.show()


def load_model_with_weights():
    # load model with pretrained weights
    return tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)

def mnist_dataloader() -> torch.utils.data.DataLoader:
    # load MNIST train data and wrapped with dataloader
    mnist_transform = transforms.Compose([
        # Convert grayscale image to 3 channels to match MobileNetV2's input
        transforms.Grayscale(num_output_channels=3),
        # Resize the image to 224x224, the standard input size for MobileNetV2
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize the tensor using ImageNet's mean and standard deviation
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    train_dataset = torchvision.datasets.MNIST(root="./data",train=False,download=False,transform=mnist_transform)
    return DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

def replace_and_retrain(model, dataloader):
    print(f"== Starting retrain the model")
    model.train()
    # freeze pretrained parameters for feature layers
    for feature_parameter in model.features.parameters():
        feature_parameter.requires_grad = False

    # retrieve the last layer of classifier
    last_classifier_layer = model.classifier[-1]
    # the in_features of last_classifier_layer
    num_features = last_classifier_layer.in_features

    # our target classifications for MINIST
    num_classes = 10
    # replace the last layer for 10 items prediction results
    model.classifier[-1] = nn.Linear(in_features=num_features, out_features=num_classes)
    print(f"After replacement:\n {model.classifier}")
    
    # training strategy, the entire layer of classifier will be retrained
    loss_function = torch.nn.CrossEntropyLoss()
    # Only optimize the parameters that require gradients
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # just several epochs
    num_epoch = 3
    for n in range(num_epoch):
        training_loss = 0.0
        # batch = 0
        for imgs, labels in dataloader:
            # batch += 1
            # print(f"batch {batch} running...")

            optimizer.zero_grad()
            ouputs = model(imgs)
            loss = loss_function(ouputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * imgs.size(0)

        epoch_loss = training_loss / len(dataloader.dataset)
        print(f"Epoch({n+1}/3):   loss: {epoch_loss}")

    return model


if __name__ == "__main__":

    show_menu()
    choose = input("Choose: ").strip()

    if choose == '1':  # Directly make predictions without any change
        # load ImageNet class name indecies
        current_dir = Path(__file__).parent
        class_names_dict = utils.load_classnames_json(str(current_dir / "./imagenet_class_index.json"))

        # load model and data
        model = load_model_with_weights()
        dataloader = mnist_dataloader()

        print("== Prediction ==")
        predict_digitals(model, dataloader.dataset, class_names_dict)

        # print(f"dataset size : {len(dataloader.dataset)}")

    elif choose == '2':  # Replace the classifier for 10 item classifications, retrained and make predictions
        # load model
        model = load_model_with_weights()
        dataloader = mnist_dataloader()
        model = replace_and_retrain(model, dataloader)
        print(f"After retraining, we make prediction again")

        print("== Prediction ==")
        predict_digitals(model, dataloader.dataset, None)

    else:
        print("terminated")
