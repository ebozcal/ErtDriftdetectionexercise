import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from scipy.special import kl_div
from torch.autograd import Variable
from sklearn.manifold import TSNE
from cifar10_models.vgg import vgg11_bn
# Device configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 4

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Dropping output layer (the ImageNet classifier)


#model = torch.load("cnn.pth").to(device)
model = vgg11_bn(pretrained=True)
model.eval() 

def slice_model(original_model, from_layer=None, to_layer=None):
    return nn.Sequential(*list(original_model.children())[from_layer:to_layer])

model = slice_model(model, to_layer=-1).to(device)

summary(model, input_size=(3, 32, 32))
#sys.exit()
embeddings = []
labelslist = []
z = 0
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images).flatten(start_dim=1)
        #print("output:", outputs.shape)
        #sys.exit()
        embeddings.extend(outputs.detach().cpu().numpy())
        labelslist.extend(labels.detach().cpu().numpy())
        z +=1
        if z>124:
            break

print(len(embeddings))
print(len(labelslist))

# Create 2 dimentions tsne ectors from the embedding ectors 
tsne = TSNE(n_components=2, random_state=42)
embedded_vectors = tsne.fit_transform(embeddings)
#print(embedded_vectors)

unique_labels = np.unique(labelslist)
num_labels = len(unique_labels)
print("number of labels:", num_labels)
label_colors = plt.cm.get_cmap('tab10', num_labels)

# Create a scatter plot of the embedded vectors with labeled points
plt.figure(figsize=(8, 8))
for i, label in enumerate(unique_labels):
    indices = labelslist == label
    #print(indices)
    plt.scatter(embedded_vectors[indices, 0], embedded_vectors[indices, 1], color=label_colors(i), label=classes[list(unique_labels).index(label)])

plt.title("t-SNE Visualization of Embedded Vectors")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.show()
