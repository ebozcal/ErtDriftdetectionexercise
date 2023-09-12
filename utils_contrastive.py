import torch
import torch.nn as nn


class TestModel:
    def __init__(self, data_loader, model, device, batch_size, classes):
        self.data_loader = data_loader
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.classes = classes
    def test_model(self):
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]
            for images, labels in self.data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
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
    
class TripletLossWithKLDivergence(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLossWithKLDivergence, self).__init__()
        self.margin = margin

    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        # Compute KL divergence between anchor and positive embeddings
        kl_div_pos = nn.functional.kl_div(
            torch.log_softmax(anchor_embeddings, dim=1),
            torch.softmax(positive_embeddings, dim=1),
            reduction='batchmean'
        )

        # Compute KL divergence between anchor and negative embeddings
        kl_div_neg = nn.functional.kl_div(
            torch.log_softmax(anchor_embeddings, dim=1),
            torch.softmax(negative_embeddings, dim=1),
            reduction='batchmean'
        )

        # Calculate triplet loss using KL divergence
        loss = torch.relu(kl_div_pos - kl_div_neg + self.margin)

        return loss.mean()