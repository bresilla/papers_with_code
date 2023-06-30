import torch
from torch import nn
import lightning.pytorch as pl
import torch.nn.functional as F


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
