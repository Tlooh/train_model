"""

datasets used to train、Val reward model, include:

* S2C_Data

* ImageReward_Data

"""


import torch
import json
from torch.utils.data import Dataset



class S2C_Dataset_BLIP(Dataset):
    def __init__(self, data_path, clip_tokenizer, blip_tokenizer):
        # load json
        with open(data_path, "r") as f:
            self.data = json.load(f)
        
        self.clip_tokenizer = clip_tokenizer
        self.blip_tokenizer = blip_tokenizer
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # dict_item: img_worse, img_better, text_ids, text_mask
        dict_item = self.handle_data(self.data[index])
        return dict_item
        
    def handle_data(self, item):
        dict_item = {}
        
        # used to CLIP encoder
        dict_item['simple_input_ids'] = self.clip_tokenizer(
            item['simple'], max_length=self.clip_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        dict_item['complex_input_ids']  = self.clip_tokenizer(
            item['complex'], max_length=self.clip_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        # blip reward 
        rm_simple_inputs = self.blip_tokenizer(
            item['simple'], max_length=35, padding="max_length", truncation=True, return_tensors="pt"
        )
        rm_complex_inputs = self.blip_tokenizer(
            item['complex'], max_length=35, padding="max_length", truncation=True, return_tensors="pt"
        )

        dict_item['rm_simple_input_ids'] = rm_simple_inputs.input_ids
        dict_item['rm_simple_mask'] = rm_simple_inputs.attention_mask

        dict_item['rm_complex_input_ids'] = rm_complex_inputs.input_ids
        dict_item['rm_complex_mask'] = rm_complex_inputs.attention_mask

        return dict_item


class S2C_Dataset_Rank(Dataset):
    def __init__(self, data_path, clip_tokenizer, blip_tokenizer, rank_list = [1, 4, 7]):
        # load json
        with open(data_path, "r") as f:
            self.data = json.load(f)

        self.rank_list = rank_list
        self.clip_tokenizer = clip_tokenizer
        self.blip_tokenizer = blip_tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dict_item = self.handle_data(self.data[index])

        return dict_item
        
    def handle_data(self, item):
        dict_item = {}
        rewards = item['rewards']
        # print(rewards)
        score_list = []

        for rank_idx in self.rank_list:
            score_list.append(rewards[rank_idx - 1])
        dict_item['score_list'] = score_list
        
        dict_item['input_ids'] = self.clip_tokenizer(item["text"], max_length=self.clip_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

        rm_inputs  = self.blip_tokenizer(item["text"], max_length=self.blip_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        dict_item['rm_input_ids'] = rm_inputs.input_ids
        dict_item['rm_mask'] = rm_inputs.attention_mask
        
        return dict_item


        
        