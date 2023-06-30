from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import lightning.pytorch as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
import torch



both_transform = A.Compose(
    [A.Resize(width=config.image_size[0], height=config.image_size[1]),], 
    additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)


class ImageDataset(Dataset):
    def __init__(self, root_dir):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.files[index])
        image = Image.open(img_path)
        width, height = image.size
        left_half = np.array(image.crop((0, 0, width // 2, height)))
        right_half = np.array(image.crop((width // 2, 0, width, height)))
        augmented = both_transform(image=left_half, image0=right_half)
        left, right = augmented["image"], augmented["image0"]
        left = transform_only_input(image=left)["image"]
        right = transform_only_mask(image=right)["image"]
        return left, right
    

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = config.data_dir, batch_size: int = config.batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit":
            self.all_data = ImageDataset(root_dir=os.path.join(self.data_dir, "train"))
            train_size = int(config.train_test_split * len(self.all_data))
            val_size = len(self.all_data) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.all_data, [train_size, val_size])
        if stage == "test":
            self.test_dataset = ImageDataset(root_dir=os.path.join(self.data_dir, "val"))

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader
    
def test():
    import matplotlib.pyplot as plt
    dataset = DataModule()
    dataset.setup("fit")
    loader = dataset.train_dataloader()
    for x, y in loader:
        print(x.shape, y.shape)
        plt.imshow(x[0].permute(1, 2, 0))
        plt.show()
        plt.imshow(y[0].permute(1, 2, 0))
        plt.show()
        break


if __name__ == "__main__":
    test()