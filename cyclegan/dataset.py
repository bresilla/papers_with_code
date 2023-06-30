import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import lightning.pytorch as pl
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import config

class CustomDataset(Dataset):
    def __init__(self, data_root: str, stage: str, transform=None):
        super(CustomDataset, self).__init__()
        self.root_A = data_root + "/" + f"{stage}A"
        self.root_B = data_root + "/" + f"{stage}B"
        self.transform = transform
        self.files_A = os.listdir(self.root_A)
        self.files_B = os.listdir(self.root_B)
        self.length_dataset = max(len(self.files_A), len(self.files_B))
        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        img_path_A = os.path.join(self.root_A, self.files_A[index % self.len_A])
        img_path_B = os.path.join(self.root_B, self.files_B[index % self.len_B])
        img_A = np.array(Image.open(img_path_A).convert("RGB"))
        img_B = np.array(Image.open(img_path_B).convert("RGB"))
        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        return img_A, img_B

class DataModule(pl.LightningDataModule):
    def __init__(self,  data_root: str, batch_size=8, num_workers=0):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = albu.Compose([
            albu.Resize(width=256, height=256),
            albu.HorizontalFlip(p=0.5),
            albu.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2()
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((256, 256)),
            # transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def setup(self, stage=None):
        if stage == "fit":
            self.all_dataset = CustomDataset(self.data_root, stage="train", transform=self.transform)
            train_size = int(0.8 * len(self.all_dataset))
            val_size = len(self.all_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(self.all_dataset, [train_size, val_size])
        if stage == "test":
            self.test_dataset = CustomDataset(self.data_root, stage="test", transform=self.transform)
        if stage == "predict":
            self.predict_dataset = CustomDataset(self.data_root, stage="test", transform=self.transform)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return val_loader
    
    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return test_loader
    
    def predict_dataloader(self):
        predict_loader = DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return predict_loader
    
def test():
    import matplotlib.pyplot as plt
    data_module = DataModule(config.data_root)
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    for x, y in train_loader:
        plt.imshow(x[0].permute(1, 2, 0))
        plt.show()
        plt.imshow(y[0].permute(1, 2, 0))
        plt.show()
        break
    val_loader = data_module.val_dataloader()
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    print(len(train_loader), len(val_loader), len(test_loader))     

if __name__ == "__main__":
    test()