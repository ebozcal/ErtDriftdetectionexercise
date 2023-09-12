import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
import sys
import logging
from utils_contrastive import TestModel, RedesineModel
from sklearn.model_selection import train_test_split

# Define your custom model for fine-tuning

#logging.basicConfig(filename='constrastive_file.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Configure logging to save messages to a file
logging.basicConfig(filename='constrastive_file.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel, self).__init__()       
        self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
         
    def forward(self, x):
        x = self.model(x)
        return x

# Load dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Initialize the model
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
embeddingmodel = EmbeddingModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
num_epochs = 1
batch_size = 128
log_interval = 10
margin = 0.5
optimizer = optim.Adam(embeddingmodel.parameters(), lr=learning_rate)
cosine_loss = nn.CosineEmbeddingLoss(margin=margin) 
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataset = torch.utils.data.Subset(dataset, list(range(0, 49152)))
# Divide the dataset in two to represent training data and inference data and prepare the DataLoader for both

#dataset_train, dataset_test = random_split(dataset, [int(len(dataset) * 0.20), int(len(dataset) * 0.80)])
#dataset_train = torch.utils.data.Subset(dataset, list(range(0, 9984)))
#dataset_train = torch.utils.data.Subset(dataset, list(range(0, 32768)))

#dataset_test = torch.utils.data.Subset(dataset, list(range(32768, 49152)))

dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=42)

dataset_train = torch.utils.data.Subset(dataset, list(range(0, 39296)))
dataset_test = torch.utils.data.Subset(dataset, list(range(0, 3728)))

data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
print("base model test result")
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
model.to(device)
testmodel = TestModel(data_loader_test, model, device, batch_size, classes)
testmodel.test_model()

  
for epoch in range(num_epochs):
    embeddingmodel.train()
    for batch_idx, (inputs, labels) in enumerate(data_loader_train):

        inputs, labels = inputs.to(device), labels.to(device)

        # Create a dictionary to organize samples by class
        class_to_samples = {i: [] for i in range(num_classes)}
        for idx, label in enumerate(labels):
            class_to_samples[label.item()].append(idx)
        anchor_indices = []
        positive_indices = []
        negative_indices = []
        #print(f'class_to_samples:{class_to_samples}')

        for class_idx, sample_indices in class_to_samples.items():
            if len(sample_indices) < 2:
                continue
            # Select positive pairs from the same class
            for i in range(len(sample_indices) - 1):
                for j in range(i + 1, len(sample_indices)):
                    
                    # Select a negative sample from a different class
                    different_classes = [c for c in class_to_samples.keys() if c != class_idx]
                    #print(f'different classes:{different_classes}')
                    negative_class = np.random.choice(different_classes)
                    #print(f'negative classes:{negative_class}')
                    
                    if negative_class in class_to_samples and class_to_samples[negative_class]:
                      
                        negative_sample = np.random.choice(class_to_samples[negative_class])
                        negative_indices.append(negative_sample)
                        anchor_indices.append(sample_indices[i])
                        positive_indices.append(sample_indices[j])
                    else:
                        print("No samples available for the negative class:", negative_class)
                        #print(f'class_to_samples:{class_to_samples}')

                    
        #print(labels)
        #print("len of anchor embedding", anchor_indices)
        #print("len of positive embedding", positive_indices)
        #print("len of negative embedding", negative_indices) 
        #sys.exit()     
    # Proceed with your further processing using negative_sample

        # Extract embeddings using the selected indices
        embeddingmodel.to(device)
        anchor_embeddings = embeddingmodel(inputs[anchor_indices]).flatten(start_dim=1)
        #print("type of anchor embedding", anchor_embeddings.shape)
        positive_embeddings = embeddingmodel(inputs[positive_indices]).flatten(start_dim=1)
        #print("len of positive embedding", len(positive_embeddings))
        negative_embeddings = embeddingmodel(inputs[negative_indices]).flatten(start_dim=1)
        #print("len of negative embedding", len(negative_embeddings))
        #print(type(negative_embeddings))
        # Compute cosine similarity loss
        target_similarity_pos = torch.ones(anchor_embeddings.shape[0]).to(device)
        target_similarity_neg = -torch.ones(anchor_embeddings.shape[0]).to(device)
        loss = cosine_loss(anchor_embeddings, positive_embeddings, target_similarity_pos) + cosine_loss(anchor_embeddings, negative_embeddings, target_similarity_neg)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list = []
        if batch_idx % log_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader_train)}], Loss: {loss.item():.4f}")
            loss_list.append(round(loss.item(), 4))
#print("ertugrul")  
#torch.save(model, './cons.pth')
#logging.info('Cosine_loss: %s', loss_list)
#logging.info(f'Cosine_loss {loss_list}')
#loaded_model = torch.load('cons.pth', map_location=torch.device('cpu'))
base_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
classification_layer = base_model.fc
redesigned_model = RedesineModel(embeddingmodel, classification_layer)
redesigned_model.to(device)
redesigned_model.eval() 
testmodel = TestModel(data_loader_test, redesigned_model, device, batch_size, classes)
testmodel.test_model()
print("Now wit the training set")
testmodel = TestModel(data_loader_train, redesigned_model, device, batch_size, classes)
testmodel.test_model()





