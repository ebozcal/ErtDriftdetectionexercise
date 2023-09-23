import os
import random
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define the root directory where your organized ImageNet dataset is located
data_dir = "/path/to/imagenet_dataset"

# Define data transformations (you can adjust these as needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean values
        std=[0.229, 0.224, 0.225]    # ImageNet standard deviations
    )
])

# Create an ImageFolder dataset for ImageNet
dataset = ImageFolder(root=data_dir, transform=transform)

# Calculate the number of samples to include in the 10% subset
subset_size = int(0.10 * len(dataset))

# Randomly select 10% of the samples
subset_indices = random.sample(range(len(dataset)), subset_size)

# Create a Subset of the dataset
subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

# Create a DataLoader for batch processing
batch_size = 32
dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

# Now, you can use the dataloader to iterate over a 10% subset of ImageNet images with labels in batches
for inputs, labels in dataloader:
    # Your training or evaluation code here
    pass
