import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import torchvision
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from dataset import CarvanaDataset, IMAGE_HEIGHT, IMAGE_WIDTH
from utils import Plotter

#randomly choose 20% of images from folder and move them to val_images folder
# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images"
TRAIN_MASK_DIR = "data/train_masks"
VAL_IMG_DIR = "data/val_images"
VAL_MASK_DIR = "data/val_masks"

accuracy = Accuracy(task="binary").to(DEVICE)

def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)
    train_loss, train_acc = 0, 0
    model.train()
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)        
        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        train_loss += loss.item()
        acc = accuracy(predictions, targets).cpu().numpy()
        train_acc += acc
        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    return train_loss / len(loader), train_acc / len(loader)


def valid_fn(loader, model, loss_fn,  folder="data/saved_images/"):
    loop = tqdm(loader)
    val_loss, val_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)
            # forward
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()
            acc = accuracy(predictions, targets).cpu().numpy()
            val_acc += acc

            masks = (torch.sigmoid(predictions) > 0.5).float()
            torchvision.utils.save_image(masks, f"{folder}/pred_{batch_idx}.png")
            torchvision.utils.save_image(targets, f"{folder}/{batch_idx}.png")

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
    return val_loss / len(loader), val_acc / len(loader)

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader = CarvanaDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=train_transform,
    )
    
    val_loader = CarvanaDataset(
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        transform=val_transforms,
    )

    train_loader = torch.utils.data.DataLoader(
        train_loader,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_loader,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    train_loss_values = []
    test_loss_values = []
    train_acc_values = []
    test_acc_values = []

    plotter = Plotter()

    for epoch in range(NUM_EPOCHS):
        # train_fn(train_loader, model, optimizer, loss_fn)
        train_loss, train_acc = train_fn(train_loader, model, optimizer, loss_fn)
        train_loss_values.append(train_loss)
        train_acc_values.append(train_acc)
        val_loss, val_acc = valid_fn(val_loader, model, loss_fn)
        test_loss_values.append(val_loss)
        test_acc_values.append(val_acc)

        plotter.plot(train_loss_values, test_loss_values, train_acc_values, test_acc_values)


if __name__ == "__main__":
    main()