# get_dataset.py: Load image data and corresponding labels

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Custom dataset class for loading image data and corresponding labels
class GetData(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file) # Read the CSV file containing labels
        self.img_dir = img_dir              # Path to the image directory
        self.transform = transform          # Transformation pipeline for image preprocessing
        
    def __len__(self):
        return len(self.labels) # Return the number of rows in the labels DataFrame
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() # Convert it to a Python list
            
        # Construct the full path to the image file using the directory and file name
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        
        # Open the image using PIL
        image = Image.open(img_name)
        
         # Extract the label from the CSV file (second column)
        label = self.labels.iloc[idx, 1]
        
        # Apply the transformation pipeline if specified
        if self.transform:
            image = self.transform(image)
            
        return image, label # Return the image and its label