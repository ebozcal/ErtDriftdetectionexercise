import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

# Load a pretrained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Preprocess input image
transform = transforms.Compose([
    transforms.Resize(256),         # Resize to 256x256
    transforms.CenterCrop(224),     # Crop the center to 224x224
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean values
        std=[0.229, 0.224, 0.225]    # ImageNet standard deviations
    )
])

# Load and preprocess the input image
image_path = "imagenet-sample-images/n01440764_tench.JPEG"
image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0)
print("image-type", type(image))  # Add batch dimension
print("image-shape", image_tensor.shape)  # Add batch dimension

# Make predictions
with torch.no_grad():
    output = model(image_tensor)

# Get the predicted class index
predicted_class_index = output.argmax().item()

print(predicted_class_index)

# Load class labels (optional)
#from torchvision import models
#with open("imagenet_classes.txt") as f:
 #   class_labels = [line.strip() for line in f.readlines()]

# Print the predicted class label
#print("Predicted class:", class_labels[predicted_class_index])
