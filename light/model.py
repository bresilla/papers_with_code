import os
import torch
from torch import nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import lightning.pytorch as pl
# from lightning.pytorch.loggers import TensorBoardLogger
import torch.nn.functional as F



class MinstDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = os.getcwd(), batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True, transform=transforms.ToTensor())
        MNIST(self.data_dir, train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage=None):
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size)
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size)
        return mnist_test

# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def training_step(self, batch, batch_idx):
        self._one_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        self._one_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        self._one_step(batch, batch_idx, "test")
    
    def _one_step(self, batch, batch_idx, stage: str):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log(f"{stage}_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder()


logger = pl.loggers.TensorBoardLogger("runs", name="my_model")
dataset = MinstDataModule("MINST", batch_size=32)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=100, logger=logger)
trainer.fit(model=autoencoder, train_dataloaders=dataset)