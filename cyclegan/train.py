import torch
import torch.nn as nn
from discriminator import Discriminator
from generator import Generator
from dataset import DataModule as MyData
import config
from tqdm import tqdm

def train_fn(dis_H, dis_Z, gen_H, gen_Z, loader, opt_D, opt_G, l1, mse):
    print(len(loader))
    pass


def main():
    dis_H = Discriminator(in_channels=3).to(config.device)
    dis_Z = Discriminator(in_channels=3).to(config.device)
    gen_H = Generator(in_channels=3, num_residuals=9).to(config.device)
    gen_Z = Generator(in_channels=3, num_residuals=9).to(config.device)
    opt_D = torch.optim.Adam(
        list(dis_H.parameters()) + list(dis_Z.parameters()),
        lr=config.learning_rate,
        betas=(0.5, 0.999)
    )
    opt_G = torch.optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr=config.learning_rate,
        betas=(0.5, 0.999)
    )
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    dataset = MyData
    dataset.setup(stage="fit")
    loader = dataset.train_dataloader()

    for epoch in range(config.num_epochs):
        train_fn(dis_H, dis_Z, gen_H, gen_Z, loader, opt_D, opt_G, l1, mse)

main()