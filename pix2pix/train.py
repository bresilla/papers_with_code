import torch
from utils import save_checkpoint, load_checkpoint, Plotter, save_some_examples
import torch.nn as nn
import torch.optim as optim
from dataset import ImageDataset, IMAGE_HEIGHT, IMAGE_WIDTH
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm
import torchvision

DATASET = "maps"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = f"data/{DATASET}/train"
VAL_DIR = f"data/{DATASET}/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

accuracy = Accuracy(task="binary").to(DEVICE)

def train_fn(epoch, loader, dis, gen, opt_dis, opt_gen, loss_fn_1, loss_fn_2):
    loop = tqdm(loader)
    train_loss, train_acc = 0, 0
    gen.train(), dis.train()
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)      
        # train discriminator
        gen_fake = gen(data)
        dis_real = dis(data, targets)
        dis_fake = dis(data, gen_fake.detach())
        dis_real_loss = loss_fn_1(dis_real, torch.ones_like(dis_real))
        dis_fake_loss = loss_fn_1(dis_fake, torch.zeros_like(dis_fake))
        loss_dis = (dis_real_loss + dis_fake_loss) / 2
        opt_dis.zero_grad()
        loss_dis.backward()
        opt_dis.step()

        # train generator
        dis_fake = dis(data, gen_fake)
        loss_gen = loss_fn_1(dis_fake, torch.ones_like(dis_fake))
        loss_l1 = loss_fn_2(gen_fake, targets) * L1_LAMBDA
        loss_gen += loss_l1
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        train_loss += (loss_dis.item() + loss_gen.item())/2
    return train_loss / len(loader)


def valid_fn(epoch, loader, gen, loss_fn, folder=f"data/{DATASET}/saved_images/"):
    #get the length of the dataset loader
    loop = tqdm(loader)
    val_loss, val_acc = 0, 0
    gen.eval()
    with torch.inference_mode():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)      
            # forward
            predictions = gen(data)
            loss = loss_fn(predictions, targets)
            predictions = predictions * 0.5 + 0.5  # remove normalization
            val_loss += loss.item()
            if batch_idx % int(len(loader)/10) == 0:
                torchvision.utils.save_image(predictions, folder + f"/{str(epoch).zfill(3)}.{batch_idx}_gen.png")
                torchvision.utils.save_image(data * 0.5 + 0.5, folder + f"/{str(epoch).zfill(3)}.{batch_idx}_0in.png")
            loop.set_postfix(loss=loss.item())
    return val_loss / len(loader)

def main():
    dis = Discriminator(in_channels=CHANNELS_IMG).to(DEVICE)
    gen = Generator(in_channels=CHANNELS_IMG).to(DEVICE)
    opt_dis = optim.Adam(dis.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion_1 = nn.BCEWithLogitsLoss()
    criterion_2 = nn.L1Loss()

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), gen, opt_gen, DEVICE)
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), dis, opt_dis, DEVICE)
    
    train_dataset = ImageDataset(root_dir=TRAIN_DIR)
    val_dataset = ImageDataset(root_dir=VAL_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    plotter = Plotter()

    for epoch in range(NUM_EPOCHS):
        train_loss = train_fn(epoch, train_loader, dis, gen, opt_dis, opt_gen, criterion_1, criterion_2)
        if SAVE_MODEL:
            checkpoint = {
                "state_dict": gen.state_dict(),
                "optimizer": opt_gen.state_dict(),
            }
            save_checkpoint(checkpoint, filename=CHECKPOINT_GEN)
            checkpoint = {
                "state_dict": dis.state_dict(),
                "optimizer": opt_dis.state_dict(),
            }
            save_checkpoint(checkpoint, filename=CHECKPOINT_DISC)
        test_loss = valid_fn(epoch, val_loader, gen, criterion_1)
        print(f"Epoch: {epoch} || Train Loss: {train_loss:.4f} || Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
