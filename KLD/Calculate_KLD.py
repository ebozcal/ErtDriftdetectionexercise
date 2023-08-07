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

logging.basicConfig(filename='my_log_file.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
batch_size = 8
# Step 1: Set up the Deep Learning Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use a pre-trained ResNet model
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()  # Set the model to evaluation mode (no gradient computation)
summary(model, input_size = (3, 32, 32))
# Step 2: Extract Embeddings
def extract_embeddings(model, dataloader):
    embeddings = []
    labelslist = []
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            outputs = model(data)
            outputs = outputs.flatten(start_dim=1) 
            embeddings.extend(outputs.detach().cpu().numpy())
            labelslist.extend(labels.detach().cpu().numpy())
    #return torch.cat(embeddings, dim=0)
    return torch.tensor(embeddings), labelslist


# Step 3: Prepare the DataLoader for both datasets
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

dataset1 = torch.utils.data.Subset(dataset, list(range(0, 10000)))
dataset2 = torch.utils.data.Subset(dataset, list(range(10000, 20000)))

data_loader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size,
                                         shuffle=False)
data_loader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size,
                                         shuffle=False)
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
image, label = next(iter(data_loader1))
#imshow(torchvision.utils.make_grid(image))

# Step 4: Implement KL Divergence Loss
def kl_divergence_loss(p, q):
    return torch.sum(p * torch.log(p / q))

# Step 5: Train the Model with KL Divergence Loss
# Extract embeddings from both datasets
embeddings_train, labels_list_train = extract_embeddings(model, data_loader1)
embeddings_new, labels_list_new= extract_embeddings(model, data_loader2)
# Normalize embeddings to unit length
embeddings_train = embeddings_train / torch.norm(embeddings_train, dim=1, keepdim=True)
embeddings_new = embeddings_new / torch.norm(embeddings_new, dim=1, keepdim=True)
 
# Compute KL divergence loss
#loss = kl_divergence_loss(embeddings1[1], embeddings2[0])
#print("KL Divergence Loss: ", loss.item())
#print("labels1_fisrt_10:", labelslist1[:10])
#print("labels2_fisrt_10:", labelslist2[:10])

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

unique_labels = np.unique(labels_list_train)
kl_loss_labels = {key:0 for key in unique_labels}

def calculate_KLD_traindata(unique_labels, embeddings_train, kl_loss_labels):
    kl_loss = []
    for label in unique_labels:
        indices = labels_list_train == label
        embedding_label = embeddings_train[indices]
        #print("embedding_shape", embedding.shape)
        for i in range(embedding_label.shape[0]):
            try:
                kl = kl_divergence_loss(embedding_label[i], embedding_label[i+1])
                #print(kl)
                if float("inf") > kl > 0:
                    kl_loss.append(kl.item())
            except:
                continue
        #print(f"len of the list for {label} is {len(kl_loss)}")
        kl_loss_labels[label] = round(np.mean(kl_loss), 2)
        kl_loss = []
    return kl_loss_labels
#print(f"maximum kl loss = {max(kl_loss)}")
#print(f"minimum kl loss = {min(kl_loss)}")
KLD_train = calculate_KLD_traindata(unique_labels, embeddings_train, kl_loss_labels)
print(KLD_train)

kl_loss_new = {key:[] for key in unique_labels}

def calculate_KLD_for_newdata(unique_labels, embeddings_train, embeddings_new, labels_list_train, labels_list_new, KLD_train, labels_dict_new):

    for label in unique_labels:
        indices1 = labels_list_train == label
        indices2 = labels_list_new == label
        embedding_label1 = embeddings_train[indices1]
        embedding_label2 = embeddings_new[indices2]
        #print("embedding_shape", embedding.shape)
        for i in range(embedding_label1.shape[0]):
            try:
                kl = kl_divergence_loss(embedding_label1[i], embedding_label2[i])
                #print(kl)
                if float("inf") > kl > 0 and kl > KLD_train[label]:
                    labels_dict_new[label].append(round(kl.item(), 2))
            except:
                continue
        #print(f"len of the list for {label} is {len(kl_loss)}")
        
    return labels_dict_new

kl_loss_new = calculate_KLD_for_newdata(unique_labels, embeddings_train, embeddings_new, labels_list_train, labels_list_new, KLD_train, kl_loss_new)
number_of_diverted = {key:0 for key in classes}
for label in kl_loss_new:
    number_of_diverted[classes[label]] = len(kl_loss_new[label])
    print(f'The number of samples diverted from the train data for class {label} is {len(kl_loss_new[label])} ')
print("number_of_diverted", number_of_diverted)

logging.info('KLD_train: %s', KLD_train)
logging.info('kl_loss_new: %s', kl_loss_new)
logging.info('number_of_diverted: %s', number_of_diverted)

