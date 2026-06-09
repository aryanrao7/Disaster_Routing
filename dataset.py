import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class DisasterDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        
        # Grab all the filenames in the directory and sort them so they match up
        self.images = sorted(os.listdir(image_dir))
        
    def __len__(self):
        # The AI engine needs to know exactly how many examples exist in an "epoch"
        return len(self.images)
        
    def __getitem__(self, index):
        # Construct the file paths for this specific image and its corresponding mask
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        # Read the image and mask from the hard drive
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Masks are just 1 channel
        
        # Resize everything to 256x256 (U-Net expects perfectly square dimensions)
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        # Neural Network Math Conversion
        # PyTorch expects images to be floats between 0.0 and 1.0 (not 0-255 pixels)
        image = image / 255.0
        mask = mask / 255.0
        
        # PyTorch expects color channels FIRST: (Channels, Height, Width)
        image = np.transpose(image, (2, 0, 1))
        
        # Convert the raw numpy arrays into PyTorch Tensors (GPU ready)
        image_tensor = torch.tensor(image, dtype=torch.float32)
        
        # Masks need an explicit channel dimension added: (1, Height, Width)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) 
        
        return image_tensor, mask_tensor

# --- Quick Test ---
if __name__ == "__main__":
    print("Dataset pipeline compiled successfully.")
    print("Next step: Feed this data into the U-Net.")
