import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchsummary import summary
import sys
import logging
import pickle


# Class for performing contrastive learning using cosine similarity to improve models perfromence on the new dataset
class ContrastiveCosine:
    def __init__(self, device, batch_size, num_epochs, learning_rate, classes, loss_function, model):
        # Initialize parameters
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.classes = classes
        self.loss_function = loss_function
        self.model = model

    # Create a submodel for extracting embeddings vectors
    def EmbeddingModel(self):
        embeddingmodel = nn.Sequential(*list(self.model.children())[:-1])
        return embeddingmodel

    # Train the contrastive learning model
    def contrastive_train(self):
        embeddingmodel = self.EmbeddingModel()
        embeddingmodel.train()
        loss_list = []
        loss_list_epoch = []
        for epoch in range(self.num_epochs):
            for batch_idx, (inputs, labels) in enumerate(data_loader_train):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Create a dictionary to organize samples by class
                class_to_samples = {i: [] for i in range(len(self.classes))}
                for idx, label in enumerate(labels):
                    class_to_samples[label.item()].append(idx)
                anchor_indices = []
                positive_indices = []
                negative_indices = []

                for class_idx, sample_indices in class_to_samples.items():
                    if len(sample_indices) < 2:
                        continue

                    # Select positive pairs from the same class
                    for i in range(len(sample_indices) - 1):
                        for j in range(i + 1, len(sample_indices)):

                            # Select a negative sample from a different class
                            different_classes = [c for c in class_to_samples.keys() if c != class_idx]
                            negative_class = np.random.choice(different_classes)

                            if negative_class in class_to_samples and class_to_samples[negative_class]:
                                negative_sample = np.random.choice(class_to_samples[negative_class])
                                negative_indices.append(negative_sample)
                                anchor_indices.append(sample_indices[i])
                                positive_indices.append(sample_indices[j])
                            else:
                                print("No samples available for the negative class:", negative_class)
                                print(f'class_to_samples:{class_to_samples}')

                # Extract embeddings using the selected indices
                embeddingmodel.to(device)
                anchor_embeddings = embeddingmodel(inputs[anchor_indices]).flatten(start_dim=1)
                positive_embeddings = embeddingmodel(inputs[positive_indices]).flatten(start_dim=1)
                negative_embeddings = embeddingmodel(inputs[negative_indices]).flatten(start_dim=1)

                # Compute cosine similarity loss
                target_similarity_pos = torch.ones(anchor_embeddings.shape[0]).to(device)
                target_similarity_neg = -torch.ones(anchor_embeddings.shape[0]).to(device)
                loss = self.loss_function(anchor_embeddings, positive_embeddings, target_similarity_pos) + self.loss_function(anchor_embeddings, negative_embeddings, target_similarity_neg)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader_train)}], Loss: {loss.item():.4f}")
                    loss_list.append(round(loss.item(), 4))
            print(f'lost_lsit:{loss_list}')
            loss_list_epoch.append(round(np.mean(loss_list), 4))
            print(f'lost_lsit_epoch:{loss_list_epoch}')
            loss_list = []
        with open ("loss_labels.pkl", "wb") as f:
            pickle.dump(loss_list_epoch, f)

        return embeddingmodel

class PlotTsne:
    def __init__(self, embeddingmodel, dataloader, device):
        self.embeddingmodel = embeddingmodel
        self.dataloader = dataloader
        self.device = device
    # Extract embeddings from datasets using the restructured model to plot t-SNE visualization
    def _extract_embeddings(self):
        embeddings = []
        labelslist = []
        with torch.no_grad():
            for data, labels in self.dataloader:
                data = data.to(self.device)
                outputs =self.embeddingmodel(data)
                outputs = outputs.flatten(start_dim=1)
                embeddings.extend(outputs.detach().cpu().numpy())
                labelslist.extend(labels.detach().cpu().numpy())

        return embeddings, labelslist
     # Define a function to normalize embeddings to unit length
    def _normalize_embeddings(self, embeddings):
        embeddings = torch.tensor(embeddings)
        embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        return embeddings
    # Plot t-SNE visualization of embedded vectors
    def _plot_tsne_embeddings(self):
        embeddings_vectors, labels = self._extract_embeddings()
        embeddings_vectors = self._normalize_embeddings(embeddings_vectors)
        tsne = TSNE(n_components=2, random_state=42)
        embedded_tsne = tsne.fit_transform(embeddings_vectors)
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)
        label_colors = plt.cm.get_cmap('tab10', num_labels)
        plt.figure(figsize=(8, 8))
        for i, label in enumerate(unique_labels):
            indices = labels == label
            plt.scatter(embedded_tsne[indices, 0], embedded_tsne[indices, 1], color=label_colors(i), label=classes[list(unique_labels).index(label)])
        plt.title("t-SNE Visualization of Embedded Vectors")
        plt.xlabel("t-SNE Component1")
        plt.ylabel("t-SNE Component2")
        plt.legend()
        plt.show()

# Subclass for the restructuring the embedding model adding classification head after contrastive training
class RedesineModel(nn.Module):
    def __init__(self, embedding_model, classification_layer):
        super(RedesineModel, self).__init__()      
        self.embedding_model = embedding_model
        self.last_layer = classification_layer

    def forward(self, x):
        features = self.embedding_model(x)
        features = features.view(features.size(0), -1)
        output = self.last_layer(features)
        return output

# Subclass for testing the redefined model
class TestModel(ContrastiveCosine):
    def __init__(self, data_loader, redisegned_model):
        super().__init__(device, batch_size, num_epochs, learning_rate, classes, loss_function, model)
        self.data_loader = data_loader
        self.redisegned_model = redisegned_model

    def test_model(self):
        with torch.no_grad():
            #self.redisegned_model.eval()
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]
            for images, labels in self.data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.redisegned_model(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for i in range(self.batch_size):
                    label = labels[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network: {acc} %')

            for i in range(10):
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {self.classes[i]}: {acc} %')

if __name__ == "__main__":

    # # Define parameters and load the base model
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.001
    num_epochs = 1
    batch_size = 128
    log_interval = 10
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    margin = 0.5
    loss_function = nn.CosineEmbeddingLoss(margin=margin)
    contrastivelr = ContrastiveCosine(device, batch_size, num_epochs, learning_rate, classes, loss_function, model)
    embeddingmodel = contrastivelr.EmbeddingModel()
    optimizer = optim.Adam(embeddingmodel.parameters(), lr=learning_rate)

    # Load and divide the dataset in two to represent training data and inference data and prepare the DataLoader for both
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataset_train = torch.utils.data.Subset(dataset, list(range(0, 32768)))
    #dataset_train = torch.utils.data.Subset(dataset, list(range(0, 1000)))

    dataset_test = torch.utils.data.Subset(dataset, list(range(32768, 49152)))
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # Testing the finetuned model
    print("xxxxxxxxxxxxTraning Startedxxxxxxxxxxxxxxxx")
    finetunedmodel = contrastivelr.contrastive_train()
    
    base_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    classification_layer = base_model.fc
    redesignedmodel = RedesineModel(finetunedmodel, classification_layer)
    redesignedmodel.to(device)
    #summary(redesignedmodel, input_size = (3, 32, 32))
    #sys.exit()
    redesignedmodel.eval()
    #summary(finetunedmodel, input_size = (3, 32, 32))
    testmodel = TestModel(data_loader_test, redesignedmodel)
    testmodel.test_model()

    # Plotting the samples using 2d tsne vectors based on the embedded vectors extracted by the finetuned model
    #tsne = PlotTsne(embeddingmodel, data_loader_train, device)
    #tsne = tsne._plot_tsne_embeddings()






