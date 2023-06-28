from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import lightning.pytorch as pl
from discriminator import Discriminator
from torch.optim.lr_scheduler import LambdaLR
from generator import Generator
import torchvision

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
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        schG = LambdaLR(opt_G, lr_lambda=gamma)
        schD = LambdaLR(opt_D, lr_lambda=gamma)
        return [opt_G, opt_D], [schD, schG]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        opt_G, opt_D = self.optimizers()

        horse, zebra = batch
        # Train Discriminators H and A
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
        self.manual_backward(D_loss)
        opt_D.step()

        # Train Generators H and A
        # adversarial loss for both generators
        D_H_fake = self.disc_H(fake_horse)
        D_A_fake = self.disc_A(fake_zebra)
        D_H_fake_loss = self.mse(D_H_fake, torch.zeros_like(D_H_fake))
        D_A_fake_loss = self.mse(D_A_fake, torch.zeros_like(D_A_fake))
        adversarial_loss = (D_H_fake_loss + D_A_fake_loss) / 2

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
        self.manual_backward(G_loss)
        opt_G.step()

    def validation_step(self, batch, batch_idx):
        horse, zebra = batch
        fake_horse, fake_zebra = self.gen_H(zebra), self.gen_A(horse)
        self.log("val_loss", self.l1(fake_horse, horse), prog_bar=True, on_epoch=True)
        self.log("val_loss", self.l1(fake_zebra, zebra), prog_bar=True, on_epoch=True)
        #log images
        self.logger.experiment.add_images("horse", fake_horse, self.current_epoch)
        self.logger.experiment.add_images("zebra", fake_zebra, self.current_epoch)
