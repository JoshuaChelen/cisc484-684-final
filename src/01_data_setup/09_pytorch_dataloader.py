import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

FINAL_DIR=os.path.join("..", "..", "data", "final_dataset")
SPLITS_DIR= os.path.join(FINAL_DIR, "splits")
MANIFEST_PATH=os.path.join(FINAL_DIR, "manifest.json")

train_transform= transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05), transforms.ToTensor()
])

eval_transforms=transforms.Compose([
    transforms.ToPILImage(), transforms.ToTensor()
])


class HAM10000NPYDataset(Dataset):
    def init(self, x_path, y_path, transform=None):
        self.X=np.load(x_path).astype(np.float32)
        self.y=np.load(y_path).astype(np.int64)
        self.transform=transform
    def len(self):
        return len(self.y)
    def getItem(self, i):
        image=self.X[i]
        label=self.y[i]
        if self.transform:
            image=self.transform(image)
        else:
            image=torch.from_numpy(image).permute(2,0,1)
        label=torch.tensor(label, dtype=torch.long)
        return image, label


def build_loaders(batch_size=32, num_workers=0):
    with open(MANIFEST_PATH, "r") as f:
        manifest=json.load(f)
    
    train_dataset=HAM10000NPYDataset(
        manifest["splits"]["train"]["X"],
        manifest["splits"],["train"]["y"],
        transform=train_transform)
    
    val_dataset=HAM10000NPYDataset(
        manifest["splits"]["val"]["X"],
        manifest["splits"][ "val"][ "y"],
        transform=eval_transforms
    )

    test_dataset=HAM10000NPYDataset(
        manifest["splits"]["test"]["X"],
        manifest["splits"][ "test"][ "y"],
        transform=eval_transforms
    )

    train_loader=DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers)
    
    val_loader=DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers)
    
    test_loader=DataLoader(test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

