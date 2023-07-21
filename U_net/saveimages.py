import os
import shutil
import sys

images_list = sorted(os.listdir("/home/ertugrul/Desktop/train/"))
masks_list= sorted(os.listdir("/home/ertugrul/Desktop/train_masks/"))
#print(sorted(masks_list)[:10])
#print(sorted(images_list)[:10])


 
source_folder = "/home/ertugrul/Desktop/train/" 
destination_folder = "/home/ertugrul/Desktop/01_DriftDetection/Unet/data/test/"

#os.makedirs(destination_folder, exist_ok=True)

def copy_selected_images(images_list, source_folder, destination_folder):
    for image_name in images_list:
        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)
        shutil.copyfile(source_path, destination_path)



#mask_list_new = list(set(mask_list_new))
#image_list_new = list(set(image_list_new))


copy_selected_images(images_list[100:150], source_folder, destination_folder)
