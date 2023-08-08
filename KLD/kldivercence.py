import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import sys 
import logging
from embeddings import extract_embeddings, average_embedding_per_class
from dataloader import DataLoader


#logging.basicConfig(filename='my_log_file.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
loader = DataLoader()
train_loader, test_loader = loader.load('imagenet-mini/val')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

# Extract embeddings from both datasets
embeddings_train, labels_list_train = extract_embeddings(model, train_loader, device)
unique_labels_train = np.unique(labels_list_train)
# Calculate average embedding for each classes 
#average_embeddings = average_embedding_per_class (unique_labels_train, labels_list_train, embeddings_train)
#print(average_embeddings.shape)
#sys.exit()
embeddings_new, labels_list_new= extract_embeddings(model, test_loader, device)
unique_labels_new = np.unique(labels_list_new)
print("embedding shape", embeddings_new.shape)
#sys.exit()
# Normalize embeddings to unit length
embeddings_train = embeddings_train / torch.norm(embeddings_train, dim=1, keepdim=True)
#average_embeddings = average_embeddings / torch.norm(average_embeddings, dim=1, keepdim=True)

embeddings_new = embeddings_new / torch.norm(embeddings_new, dim=1, keepdim=True)
 
# Define a function to calculate the KL divercence loss between 2 embedding vectors
def kl_divergence_loss(p, q):
    return torch.sum(p * torch.log(p / q))
# Create a empty dictonary with each class as a key and 0 as value to collect per samples KL loss with the
# the same class members
kl_loss_train_labels = {key:0 for key in unique_labels_train}

# Calculate training data average KL loss for per class comparing each class sample with every other class samples and 
# taking the average in the end and collect them in the dictinoary
def calculate_KLD_traindata(unique_labels_train, labels_list_train, embeddings_train, kl_loss_train_labels):
    kl_loss = []
    for label in unique_labels_train:
        mask_train = labels_list_train == label
        embedding_label = embeddings_train[mask_train]
        for i in range(embedding_label.shape[0]):
            try:
                kl = kl_divergence_loss(embedding_label[i], embedding_label[i+1])
                if float("inf") > kl > 0:
                    kl_loss.append(kl.item())
            except:
                continue
        kl_loss_train_labels[label] = round(np.mean(kl_loss), 2)
        kl_loss = []
    return kl_loss_train_labels

KLD_train = calculate_KLD_traindata(unique_labels_train, labels_list_train, embeddings_train, kl_loss_train_labels)
#print(KLD_train)

# Create a empty dictonary with each class as a key and empty list as value to collect per samples KL loss with the
# the same class members
kl_loss_new = {key:[] for key in unique_labels_train}

# Calculate the new data KL loss per sample and put them in their corresponding class in the dictinoary as a list
# Here, to calculate every samples KL loss, each new sample loss computed with the every train sample loss if they 
# are from
def calculate_KLD_for_newdata(unique_labels_new, embeddings_train, embeddings_new, labels_list_train, labels_list_new, 
                              KLD_train, labels_dict_new):

    for label in unique_labels_new:
        mask_new = labels_list_new == label
        mask_train = labels_list_train == label
        embedding_new_label = embeddings_new[mask_new]
        embedding_train_label = embeddings_train[mask_train]
        for i in range(embedding_new_label.shape[0]):
            for j in embedding_train_label.shape[0]:
                try:
                    kl = kl_divergence_loss(embedding_new_label[i], embedding_train_label[j])
                    if float("inf") > kl > 0 and kl > KLD_train[label]:
                        labels_dict_new[label].append(round(kl.item(), 2))
                except:
                    continue       
    return labels_dict_new


kl_loss_new = calculate_KLD_for_newdata(unique_labels_new, embeddings_train, embeddings_new, labels_list_train, 
                                        labels_list_new, KLD_train, kl_loss_new)
number_of_diverted = {key:0 for key in unique_labels_new}

# Calculate how many new dataset's samlples are diverted from the avarage training samples KL loss by comparing 
# avarage tarining KL with new data KL divercenge from avarege value
for label in kl_loss_new:
    number_of_diverted[unique_labels_new[label]] = len(kl_loss_new[label])
    print(f'The number of samples diverted from the train data for class {label} is {len(kl_loss_new[label])} ')
print("number_of_diverted", number_of_diverted)

#logging.info('KLD_train: %s', KLD_train)
#logging.info('kl_loss_new: %s', kl_loss_new)
#logging.info('number_of_diverted: %s', number_of_diverted)

