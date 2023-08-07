
import torch
import torchvision
from dataloader import DataLoader


data_loader = DataLoader()
train_loader, test_loader = data_loader.load('imagenet-mini/train')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet152(pretrained=True)
model = model.to(device)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()    

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    #for i in range(10):
     #   acc = 100.0 * n_class_correct[i] / n_class_samples[i]
      #  print(f'Accuracy of {classes[i]}: {acc} %')



