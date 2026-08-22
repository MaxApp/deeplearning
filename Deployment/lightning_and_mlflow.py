import logging
import os

import lightning.pytorch as pl
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch.callbacks import Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import Accuracy

torch.set_float32_matmul_precision('medium')
logging.getLogger("mlflow").setLevel(logging.ERROR)


class CIFAR10DataModule(pl.LightningDataModule):
    """CIFAR10 dataset"""

    def __init__(self, data_dir='./CIFAR10_data', batch_size=64, num_workers=2):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # transformations for training data
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                               std=(0.2023, 0.1994, 0.2010)),
        ])
        
        # transformations for validation data
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                               std=(0.2023, 0.1994, 0.2010)),
        ])
        
        # CIFAR-10 labels
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck')

    def prepare_data(self):
        """Downloads the CIFAR10 dataset if not exist"""
        if os.path.exists(self.data_dir) and os.path.isdir(self.data_dir):
            print("Loading from local")
        else:
            print("Downloading data")
            
        # download the dataset, will skip if already exists
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self):
        """
        train/val datasets
        """
        # training dataset
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, transform=self.transform_train
        )
        
        # validation dataset
        self.val_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, transform=self.transform_test
        )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

class SimpleCNN(pl.LightningModule):
    
    def __init__(self, learning_rate=0.001):
        super().__init__()
        # auto save input parameters of __init__
        # in this case, that's `learning_rate`
        # so we can get from `self.hparams.learning_rate` in optimizer
        self.save_hyperparameters()
        
        # model architecture
        self.model = nn.Sequential(
            # conv layers
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # fc layers
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 10)
        )
        
        # loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        one step for training
        """
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        
        # accuracy
        preds = torch.argmax(outputs, dim=1)
        self.train_accuracy(preds, labels)
        
        # metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        one step for validation
        """
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        
        # accuracy
        preds = torch.argmax(outputs, dim=1)
        self.val_accuracy(preds, labels)
        
        # metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        # get `learning_rate` from benefit of `self.save_hyperparameters()`
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

class MLflowLoggingCallback(Callback):
    """
    Custom lightning callback to log with MLflow
    """
    
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        self.best_accuracy = 0
        self.model_save_dir = "./best_model"
        os.makedirs(self.model_save_dir, exist_ok=True)
    
    def on_train_start(self, trainer, pl_module):
        # hyperparameters
        mlflow.log_param("model_type", "SimpleCNN")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("initial_lr", pl_module.hparams.learning_rate)
        mlflow.log_param("scheduler", "ReduceLROnPlateau")
        mlflow.log_param("batch_size", trainer.datamodule.batch_size)
        mlflow.log_param("random_seed", RANDOM_SEED)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # skip if in sanity checking mode
        if trainer.sanity_checking:
            return
        
        metrics = trainer.callback_metrics
        current_epoch = trainer.current_epoch
        
        # logs
        if "train_loss" in metrics:
            mlflow.log_metric("train_loss", metrics["train_loss"].item(), step=current_epoch)
        if "val_loss" in metrics:
            mlflow.log_metric("val_loss", metrics["val_loss"].item(), step=current_epoch)
        if "val_acc" in metrics:
            accuracy = metrics["val_acc"].item() * 100
            mlflow.log_metric("accuracy", accuracy, step=current_epoch)
            
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            mlflow.log_metric("learning_rate", current_lr, step=current_epoch)
            
            # checkpoint
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                
                checkpoint = {
                    'epoch': current_epoch + 1,
                    'model_state_dict': pl_module.state_dict(),
                    'optimizer_state_dict': trainer.optimizers[0].state_dict(),
                    'val_loss': metrics["val_loss"].item(),
                    'accuracy': accuracy,
                    'random_seed': RANDOM_SEED
                }
                
                checkpoint_filename = f'best_model_checkpoint_epoch_{current_epoch + 1}.pt'
                checkpoint_path = os.path.join(self.model_save_dir, checkpoint_filename)
                torch.save(checkpoint, checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
    
    def on_train_end(self, trainer, pl_module):
        # log best accuracy
        mlflow.log_metric("best_accuracy", self.best_accuracy)
        
        pl_module.eval()

        # log the trained model
        input_example_tensor, _ = next(iter(trainer.val_dataloaders))
        input_example_numpy = input_example_tensor.cpu().numpy()
        
        pl_module.to("cpu")
        
        mlflow.pytorch.log_model(
            pytorch_model=pl_module,
            artifact_path="cifar10_cnn_model_final",
            input_example=input_example_numpy
        )
        
        print(f'Training finished. Best accuracy: {self.best_accuracy:.2f}%')

    
if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')
    logging.getLogger("mlflow").setLevel(logging.ERROR)

    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)

    data_module = CIFAR10DataModule(data_dir='./CIFAR10_data')

    mlflow.set_experiment("CIFAR10_CNN")

    # MLFlow context
    with mlflow.start_run() as run:
        
        num_epochs = 3
        model = SimpleCNN(learning_rate=0.001)

        mlflow_callback = MLflowLoggingCallback(classes=data_module.classes)
        
        # lightning Trainer
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="auto",
            devices=1,
            logger=False,  # disable default logging
            callbacks=[mlflow_callback],
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=False  # handle it in custom callback
        )
        
        # training
        trainer.fit(model, data_module)
        print(f'MLflow run id: {run.info.run_id}')