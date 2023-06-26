import inspect
import os
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_some_examples(gen, val_loader, epoch, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    x, y = next(iter(val_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class Plotter():
    def __init__(self):
        # plt.ion()
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()
        self.epoch_count = 0

    def plot(self, train_loss_values: list, test_loss_values: list, train_acc_values: list = None, test_acc_values: list = None):
        self.ax1.clear()
        self.ax2.clear()      
        self.ax1.set_xlabel("Epochs")
        self.ax1.set_title("Loss and Accuracy")

        self.ax1.plot(range(self.epoch_count), train_loss_values, label="train_loss", color='red')
        self.ax1.plot(range(self.epoch_count), test_loss_values, label="test_loss", color='darkred')
        self.ax1.tick_params(axis='y', colors='red')
        self.ax1.legend(loc='lower left')
        if train_acc_values is not None and test_acc_values is not None:
            self.ax2.plot(range(self.epoch_count), train_acc_values, label="train_acc", color='green')
            self.ax2.plot(range(self.epoch_count), test_acc_values, label="test_acc", color='darkgreen')
            self.ax2.tick_params(axis='y', colors='green')
            self.ax2.legend(loc='upper left')
        self.epoch_count += 1
        plt.draw()
        plt.pause(0.1)
