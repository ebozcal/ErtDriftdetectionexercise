import numpy as np 
import pandas as pd
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optims
import torchvision.transforms as transforms
from PIL import Image
import os
import sys


class DataLoader:
    def __init__(self):
        
        self.transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def resize_image(src, size=(128, 128), bgc="white"):
        src.thumbnail(size, Image.ANTIALIAS)
        new_image = Image.new("RGB", size, bgc)
        new_image.paste(src, (int((size[0]-src.size[0]) / 2)), int((size[1] - src.size[1]) / 2))
    
        return new_image

    def load(self, path):
        dataset = tv.datasets.ImageFolder(root=path, transform = self.transform)
        
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=32,
            num_workers=0,
            shuffle=False
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=32,
            num_workers=0,
            shuffle=False
        )

        return train_loader, test_loader

