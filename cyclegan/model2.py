import torch
import torch.nn as nn
import lightning.pytorch as pl
from discriminator import Discriminator
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from generator import Generator
import itertools

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
        # generator pair
        self.genX = Generator(img_channels=3, num_residuals=9)
        self.genY = Generator(img_channels=3, num_residuals=9)
        self.lm = 10.0
        
        # discriminator pair
        self.disX = Discriminator(in_channels=3)
        self.disY = Discriminator(in_channels=3)

        self.l1 = nn.L1Loss()


    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        optG = torch.optim.Adam(
            itertools.chain(self.genX.parameters(), self.genY.parameters()),
            lr=lr, betas=(0.5, 0.999))
        
        optD = torch.optim.Adam(
            itertools.chain(self.disX.parameters(), self.disY.parameters()),
            lr=lr, betas=(0.5, 0.999))
        
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        schG = LambdaLR(optG, lr_lambda=gamma)
        schD = LambdaLR(optD, lr_lambda=gamma)
        return [optG, optD], [schG, schD]

    def get_mse_loss(self, predictions, label):
        if label.lower() == 'real':
            target = torch.ones_like(predictions)
        else:
            target = torch.zeros_like(predictions)
        return F.mse_loss(predictions, target)
    
    def set_requires_grad(self, nets, requires_grad):
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def generator_training_step(self, imgA, imgB, opt):        
        fakeB = self.genX(imgA)
        cycledA = self.genY(fakeB)
        fakeA = self.genY(imgB)
        cycledB = self.genX(fakeA)
        sameB = self.genX(imgB)
        sameA = self.genY(imgA)
        # generator genX must fool discrim disY so label is real = 1
        predFakeB = self.disY(fakeB)
        mseGenB = self.get_mse_loss(predFakeB, 'real')
        # generator genY must fool discrim disX so label is real
        predFakeA = self.disX(fakeA)
        mseGenA = self.get_mse_loss(predFakeA, 'real')
        # compute extra losses
        identityLoss = F.l1_loss(sameA, imgA) + F.l1_loss(sameB, imgB)
        # compute cycleLosses
        cycleLoss = F.l1_loss(cycledA, imgA) + F.l1_loss(cycledB, imgB)
        # gather all losses
        extraLoss = cycleLoss + 0.5 * identityLoss
        self.genLoss = mseGenA + mseGenB + self.lm * extraLoss
        self.log('gen_loss', self.genLoss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # store detached generated images
        self.fakeA = fakeA.detach()
        self.fakeB = fakeB.detach()
        self.manual_backward(self.genLoss)
        opt.step()
        return self.genLoss

    def discriminator_training_step(self, imgA, imgB, opt):
        fakeA = self.fakeA
        fakeB = self.fakeB
        # disX checks for domain A photos
        predRealA = self.disX(imgA)
        mseRealA = self.get_mse_loss(predRealA, 'real')
        predFakeA = self.disX(fakeA)
        mseFakeA = self.get_mse_loss(predFakeA, 'fake')
        # disY checks for domain B photos
        predRealB = self.disY(imgB)
        mseRealB = self.get_mse_loss(predRealB, 'real')
        predFakeB = self.disY(fakeB)
        mseFakeB = self.get_mse_loss(predFakeB, 'fake')
        # gather all losses
        self.disLoss = 0.5 * (mseFakeA + mseRealA + mseFakeB + mseRealB)
        self.log('dis_loss', self.disLoss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.manual_backward(self.disLoss)
        opt.step()
        return self.disLoss

    def training_step(self, batch, batch_idx):
        imgA, imgB = batch
        opt_G, opt_D = self.optimizers()
        self.generator_training_step(imgA, imgB, opt_G)
        self.discriminator_training_step(imgA, imgB, opt_D)

    def validation_step(self, batch, batch_idx):
        imgA, imgB = batch
        fakeB = self.genX(imgA)
        fakeA = self.genY(imgB)
        self.log('val_loss', self.l1(imgA, fakeA), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', self.l1(imgB, fakeB), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #log images
        self.logger.experiment.add_images('fakeA', fakeA, self.current_epoch)
        self.logger.experiment.add_images('fakeB', fakeB, self.current_epoch)
        self.logger.experiment.add_images('realA', imgA, self.current_epoch)
        self.logger.experiment.add_images('realB', imgB, self.current_epoch)