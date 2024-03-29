import os
import torch
import json
from torch.utils.data import Dataset
from PIL import Image


# Step 1: json to generate images
class Gen_Dataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


# Step 2: Eval imgs generated by different model
    # model_list: imgs_dir per model
class Eval_Dataset(Dataset):
    def __init__(self, data_path, model_list):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.model_list = model_list

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        id = item['id']
        text = item['prompt']
        imgs = [os.path.join(model_dir, f"{id}.png") for model_dir in self.model_list]

        return id, text, imgs
    

