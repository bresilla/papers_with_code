import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally

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

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask==255.0] = 1.0
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask
    

def donwload_dataset(dataset):
    import opendatasets as od
    dataset = "https://www.kaggle.com/competitions/carvana-image-masking-challenge"
    data_dir = './data'
    od.download(dataset, data_dir)

def move_files(source_image="data/train_images", 
               dest_images="data/val_images", 
               source_mask="data/train_masks", 
               dest_mask="data/val_masks", 
               percentage=0.2):
    import random
    import shutil
    files = os.listdir(source_image)
    no_of_files = len(files)
    no_of_files_to_move = int(no_of_files * percentage)
    random.seed(42)
    files_to_move = random.sample(files, no_of_files_to_move)
    if not os.path.exists(dest_images):
        os.makedirs(dest_images)
    if not os.path.exists(dest_mask):
        os.makedirs(dest_mask)

    for file in files_to_move:
        shutil.move(os.path.join(source_image, file), dest_images)
        shutil.move(os.path.join(source_mask, file.replace(".jpg", "_mask.gif")), dest_mask)


if __name__ == "__main__":
    donwload_dataset()
    move_files()