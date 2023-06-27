from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import lightning.pytorch as pl
from discriminator import Discriminator
from generator import Generator

class CycleGAN(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 2e-4,
        lambda_identity: float = 0.0,
        lambda_cycle: float = 10.0,
        lambda_adversarial: float = 1.0,
        lambda_pixel: float = 100.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.gen_H = Generator(img_channels=3, num_residuals=9)
        self.gen_A = Generator(img_channels=3, num_residuals=9)
        self.disc_H = Discriminator(in_channels=3)
        self.disc_A = Discriminator(in_channels=3)

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
   
    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt_G = torch.optim.Adam(
            list(self.gen_H.parameters()) + list(self.gen_A.parameters()),
            lr=lr,
            betas=(0.5, 0.999),
        )
        opt_D = torch.optim.Adam(
            list(self.disc_H.parameters()) + list(self.disc_A.parameters()),
            lr=lr,
            betas=(0.5, 0.999),
        )
        return [opt_D, opt_G], []

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        horse, zebra = batch

        # Train Discriminators H and A
        if optimizer_idx == 0:
            # generate fakes
            fake_horse, fake_zebra = self.gen_H(zebra), self.gen_A(horse)

            # train on fake
            D_H_fake = self.disc_H(fake_horse.detach())
            D_A_fake = self.disc_A(fake_zebra.detach())
            D_H_fake_loss = self.mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_A_fake_loss = self.mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_fake_loss = (D_H_fake_loss + D_A_fake_loss) / 2

            # train on real
            D_H_real = self.disc_H(horse)
            D_A_real = self.disc_A(zebra)
            D_H_real_loss = self.mse(D_H_real, torch.ones_like(D_H_real))
            D_A_real_loss = self.mse(D_A_real, torch.ones_like(D_A_real))
            D_real_loss = (D_H_real_loss + D_A_real_loss) / 2

            # combine losses
            D_loss = (D_real_loss + D_fake_loss) / 2
            self.log("D_loss", D_loss, prog_bar=True, on_epoch=True)
            return D_loss

        # Train Generators H and A
        if optimizer_idx == 1:
            # adversarial loss for both generators
            D_H_fake = self.disc_H(fake_horse)
            D_A_fake = self.disc_A(fake_zebra)
            adversarial_loss = (
                self.mse(D_H_fake, torch.ones_like(D_H_fake))
                + self.mse(D_A_fake, torch.ones_like(D_A_fake))
            ) / 2

            # identity loss
            id_horse = self.gen_H(horse)
            id_zebra = self.gen_A(zebra)
            id_horse_loss = self.l1(id_horse, horse)
            id_zebra_loss = self.l1(id_zebra, zebra)
            identity_loss = (id_horse_loss + id_zebra_loss) / 2

            # cycle loss
            cycle_horse = self.gen_H(fake_zebra)
            cycle_zebra = self.gen_A(fake_horse)
            cycle_horse_loss = self.l1(cycle_horse, horse)
            cycle_zebra_loss = self.l1(cycle_zebra, zebra)
            cycle_loss = (cycle_horse_loss + cycle_zebra_loss) / 2

            # pixel loss
            pixel_horse_loss = self.l1(fake_horse, horse)
            pixel_zebra_loss = self.l1(fake_zebra, zebra)
            pixel_loss = (pixel_horse_loss + pixel_zebra_loss) / 2

            # combine losses
            G_loss = (
                self.hparams.lambda_adversarial * adversarial_loss
                + self.hparams.lambda_identity * identity_loss
                + self.hparams.lambda_cycle * cycle_loss
                + self.hparams.lambda_pixel * pixel_loss
            )
            self.log("G_loss", G_loss, prog_bar=True, on_epoch=True)
            return G_loss
        
    def validation_step
    
    # def on_train_epoch_end(self, outputs):
    #     horse, zebra = next(iter(self.train_dataloader()))
    #     fake_horse, fake_zebra = self.gen_H(zebra), self.gen_A(horse)
    #     img_grid = torch.cat((horse, fake_horse, zebra, fake_zebra), dim=0)
    #     self.logger.experiment.add_image("Generated Images", img_grid, self.current_epoch)
    
    # def validation_step(self, batch, batch_idx):
    #     horse, zebra = batch
    #     fake_horse, fake_zebra = self.gen_H(zebra), self.gen_A(horse)
    #     img_grid = torch.cat((horse, fake_horse, zebra, fake_zebra), dim=0)
    #     self.logger.experiment.add_image("Generated Images", img_grid, self.current_epoch)
    #     return img_grid
        
    # def test_step(self, batch, batch_idx):
    #     horse, zebra = batch
    #     fake_horse, fake_zebra = self.gen_H(zebra), self.gen_A(horse)
    #     img_grid = torch.cat((horse, fake_horse, zebra, fake_zebra), dim=0)
    #     self.logger.experiment.add_image("Generated Images", img_grid, self.current_epoch)
    #     return img_grid

    # def predict_step(self, batch, batch_idx, dataloader_idx=None):
    #     horse, zebra = batch
    #     fake_horse, fake_zebra = self.gen_H(zebra), self.gen_A(horse)
    #     img_grid = torch.cat((horse, fake_horse, zebra, fake_zebra), dim=0)
    #     self.logger.experiment.add_image("Generated Images", img_grid, self.current_epoch)
    #     return img_grid