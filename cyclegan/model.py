import torch
import torch.nn as nn
import lightning.pytorch as pl
from discriminator import Discriminator
from torch.optim.lr_scheduler import LambdaLR
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

        self.gen_H = Generator(img_channels=3, num_residuals=9)
        self.gen_A = Generator(img_channels=3, num_residuals=9)
        self.dis_H = Discriminator(in_channels=3)
        self.dis_A = Discriminator(in_channels=3)

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
   
    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt_G = torch.optim.Adam(
            list(self.gen_H.parameters()) + list(self.gen_A.parameters()),
            lr=lr,
            betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(
            list(self.dis_H.parameters()) + list(self.dis_A.parameters()),
            lr=lr,
            betas=(0.5, 0.999))
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
        D_H_fake = self.dis_H(fake_horse.detach())
        D_A_fake = self.dis_A(fake_zebra.detach())
        D_H_fake_loss = self.mse(D_H_fake, torch.zeros_like(D_H_fake))
        D_A_fake_loss = self.mse(D_A_fake, torch.zeros_like(D_A_fake))
        D_fake_loss = (D_H_fake_loss + D_A_fake_loss) / 2

        # train on real
        D_H_real = self.dis_H(horse)
        D_A_real = self.dis_A(zebra)
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
        D_H_fake = self.dis_H(fake_horse)
        D_A_fake = self.dis_A(fake_zebra)
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

class CycleGAN_2(pl.LightningModule):
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
        opt_G = torch.optim.Adam(
            itertools.chain(self.genX.parameters(), self.genY.parameters()),
            lr=lr, 
            betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(
            itertools.chain(self.disX.parameters(), self.disY.parameters()),
            lr=lr, 
            betas=(0.5, 0.999))
        
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        schG = LambdaLR(opt_G, lr_lambda=gamma)
        schD = LambdaLR(opt_D, lr_lambda=gamma)
        return [opt_G, opt_D], [schG, schD]

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