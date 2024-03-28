# age_dataset.py
from PIL import Image
import numpy as np
import torch
import os
from torch.utils.data import Dataset

class AgeDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.data = []
        for age in os.listdir(directory):
            if age != '.DS_Store':
                new_dir = os.path.join(directory, age)
                for img in os.listdir(new_dir):
                    if img.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(new_dir, img)
                        self.data.append((img_path, int(age)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, age = self.data[idx]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')


        # Convert the image to a numpy array and normalize the pixel values
        img_array = np.array(img) / 255.0
        # Transpose the array to have the channel dimension first
        img_array = img_array.transpose((2, 0, 1))
        # Convert the numpy array to a torch tensor
        img_tensor = torch.from_numpy(img_array).float()
        return img_tensor, age

        #img_array = np.array(img)
        #img_array = img_array / 255.0
        #img_tensor = torch.from_numpy(img_array)
        #return img_tensor, age
