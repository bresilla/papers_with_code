from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally

both_transform = A.Compose(
    [A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),], additional_targets={"image0": "image"},
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
    def __init__(self, root_dir, transform=None):
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