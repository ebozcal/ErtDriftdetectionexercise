import torch
import torch.nn as nn
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
#from torch.optim import optim
from model import U_net
from utils import (load_checkpoint, save_checkpoint,
                   get_loaders, check_accuracy,
                  save_predictions_as_images)

device = "cuda" if torch.cuda.is_available() else "cpu"
#hyperparameters
num_workers = 2
batch_size = 16
learning_rate = 0.001
epoch_number = 5
image_height = 160
image_width = 240
pin_memory = True
load_model = True
train_image_dir = "/content/drive/My Drive/001_Shell/train/"
train_mask_dir = "/content/drive/My Drive/001_Shell/train_masks/"
val_image_dir = "/content/drive/My Drive/001_Shell/test/"
val_mask_dir = "/content/drive/My Drive/001_Shell/test_masks/"

#def train(data_loader, model, optimizer, criterion, scaler):
def train(data_loader, model, optimizer, criterion, scaler):

    loop = tqdm(data_loader)

    for idx, (data, labels) in enumerate(data_loader):
        data = data.to(device=device)
        labels = labels.float().unsqueeze(1).to(device=device)
        #forward
        with torch.cuda.amp.autocast():
            prediction = model(data)
            loss = criterion(prediction, labels)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        #loss.backward()
        scaler.step(optimizer)
        #optimizer.step()
        scaler.update()
             
        #update tqdm loop
        #loop.set_postfix(loss==loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height = image_height, width = image_width),
            A.Rotate(limit = 35, p = 1.0),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.1), 
            A.Normalize(
                    mean = [0.0, 0.0, 0.0],
                    std = [1.0, 1.0, 1.0],
                    max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = A.Compose(
        [
            A.Resize(height = image_height, width = image_width), 
            A.Normalize(
                    mean = [0.0, 0.0, 0.0],
                    std = [1.0, 1.0, 1.0],
                    max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )
    model = U_net(in_channels = 3, out_channels = 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    train_loader, val_loader = get_loaders(train_image_dir,
                    train_mask_dir, val_image_dir, 
                    val_mask_dir, batch_size, 
                    train_transform, val_transforms,
                    num_workers, pin_memory)
    
    #if load_model:
     #   load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=device)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epoch_number):
        #train(train_loader, model, optimizer, criterion, scaler)
        train(train_loader, model, optimizer, criterion, scaler)


        #save model
        checkpoint = {
        "state_dict" :model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        #check accuracy
        check_accuracy(val_loader, model, device=device)
        #print some example to a folder
        save_predictions_as_images(val_loader, model, folder= "/content/drive/My Drive/001_Shell/saved_images/", device = device)


if __name__ == "__main__":
    main()
