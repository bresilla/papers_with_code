import torch
import torch.nn as nn
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator
from torch.optim.lr_scheduler import LambdaLR
import torchvision
import lightning as pl
import config
import os

class PixModel(pl.LightningModule):
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
        self.counter = 0
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.gen = Generator(in_channels=config.in_channels)
        self.dis = Discriminator(in_channels=config.in_channels)
        self.criterion_1 = nn.BCEWithLogitsLoss()
        self.criterion_2 = nn.L1Loss()

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt_D = optim.Adam(
            self.dis.parameters(), 
            lr=config.learning_rate, 
            betas=(0.5, 0.999))
        opt_G = optim.Adam(
            self.gen.parameters(), 
            lr=config.learning_rate, 
            betas=(0.5, 0.999))
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        schG = LambdaLR(opt_G, lr_lambda=gamma)
        schD = LambdaLR(opt_D, lr_lambda=gamma)
        return [opt_G, opt_D], [schD, schG]
    
    def train_discriminator(self, image, real, fake):
        # train discriminator
        dis_real = self.dis(image, real)
        dis_fake = self.dis(image, fake.detach())
        dis_real_loss = self.criterion_1(dis_real, torch.ones_like(dis_real))
        dis_fake_loss = self.criterion_1(dis_fake, torch.zeros_like(dis_fake))
        loss_dis = (dis_real_loss + dis_fake_loss) / 2
        self.log('loss_dis', loss_dis, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_dis
    
    def train_generator(self, image, real, fake):
        # train generator
        dis_fake = self.dis(image, fake)
        loss_gen = self.criterion_1(dis_fake, torch.ones_like(dis_fake))
        loss_l1 = self.criterion_2(fake, real) * config.l1_lambda
        loss_gen_total = loss_gen + loss_l1
        self.log('loss_gen', loss_gen_total, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_gen_total

    def training_step(self, batch):
        image, real = batch
        fake = self.gen(image)
        # train discriminator
        dis_real = self.dis(image, real)
        dis_fake = self.dis(image, fake.detach())
        dis_real_loss = self.criterion_1(dis_real, torch.ones_like(dis_real))
        dis_fake_loss = self.criterion_1(dis_fake, torch.zeros_like(dis_fake))
        self.loss_dis = (dis_real_loss + dis_fake_loss) / 2
        self.log('loss_dis', self.loss_dis, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.optimizers()[0].zero_grad()
        self.manual_backward(self.loss_dis)
        self.optimizers()[0].step()

        # train generator
        dis_fake = self.dis(image, fake)
        loss_gen = self.criterion_1(dis_fake, torch.ones_like(dis_fake))
        loss_l1 =  self.criterion_2(fake, real) * config.l1_lambda
        self.loss_gen = loss_gen + loss_l1
        self.log('loss_gen', self.loss_gen, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.optimizers()[1].zero_grad()
        self.manual_backward(self.loss_gen)
        self.optimizers()[1].step()

    def validation_step(self, batch, batch_idx):
        self.counter += 1
        with torch.inference_mode():
            data, targets = batch
            predictions = self.gen(data)
            loss = self.criterion_2(predictions, targets)
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            #log images every few steps
            if batch_idx % 10 == 0:
                #check if folder exists
                if not os.path.exists(config.save_images):
                    os.makedirs(config.save_images)
                torchvision.utils.save_image(
                    predictions, 
                    config.save_images + f"/{str(self.counter).zfill(3)}.{batch_idx}_gen.png"
                )
                torchvision.utils.save_image(
                    targets, 
                    config.save_images + f"/{str(self.counter).zfill(3)}.{batch_idx}_0in.png"
                )