
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import sys 
import logging
from dataloader import DataLoader

# Extract Embeddings
def extract_embeddings(model, dataloader, device):
    embeddings = []
    labelslist = []
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            outputs = model(data)
            outputs = outputs.flatten(start_dim=1) 
            embeddings.extend(outputs.detach().cpu().numpy())
            labelslist.extend(labels.detach().cpu().numpy())
      
    return torch.tensor(embeddings), labelslist

import torch

# Sample 2D tensor (replace this with your actual data)
tensor_data = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0]], dtype=torch.float32)

# Sample class labels (replace this with your actual labels)
class_labels = torch.tensor([0, 1, 0, 1])

# Calculate class averages
unique_classes = torch.unique(class_labels)

def average_embedding_per_class(unique_classes, class_labels, embeddings):
    class_averages = []
    for class_label in unique_classes:
        mask = class_labels == class_label
        class_data = embeddings[mask]
        class_average = class_data.mean(dim=0)
        class_averages.append(class_average)
    stacked_class_averages = torch.stack(class_averages)
    return stacked_class_averages


