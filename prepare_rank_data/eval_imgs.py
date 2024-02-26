import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

import sys
sys.path.append('/home/linhaojia/liutao/train_model')
import json
import torch 
import argparse

import random
from PIL import Image
from tqdm.auto import tqdm
from dataset import *

from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

from models.ImageReward import ImageReward_load
from models.CLIPReward import CLIPReward_load
import pdb

class Custome_Dataset(Dataset):
    def __init__(self, data_path):
        # load json
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dict_item = self.data[index]

        idx, text, generations = dict_item["id"], dict_item["text"], dict_item["generations"]

        return idx, text, generations
        

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument(
        "--json_path",
        default="/data/liutao/rank_images/json/refl_w_path.json",
        type=str,
        help="Json path to Eval Model",
    )
    parser.add_argument(
        "--benchmark",
        default="ImageReward",
        type=str,
        help="ImageReward, Aesthetic, BLIP or CLIP, CLIP_v1, CLIP_v2, splitted with comma(,) if there are multiple benchmarks.",
    )
    parser.add_argument(
        "--rm_path",
        default="/data/liutao/checkpoints/ImageReward/ImageReward.pt",
        type=str,
        help="Path to place downloaded reward model in.",
    )
    parser.add_argument(
        "--gpu_id",
        default=7,
        type=str,
        help="GPU ID(s) to use for CUDA.",
    )


    args = parser.parse_args()
    args.filename = os.path.basename(__file__)

    return args


def sorted_by_rewards(json_path):
    # 使用 json.load 从文件中加载 JSON 数据
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # 遍历每个项并对其 "rewards" 进行排序
    for item in data:
        # 获取排序后的索引
        sorted_indices = sorted(range(len(item['rewards'])), key=lambda k: item['rewards'][k], reverse=True)
        # print(sorted_indices)
        # 根据排序后的索引获取排序后的 "generations" 和 "rewards"
        sorted_generations = [item['generations'][i] for i in sorted_indices]
        sorted_rewards = [item['rewards'][i] for i in sorted_indices]

        item["generations"] = sorted_generations
        item['reanking'] = [i+1 for i in range(len(sorted_rewards))]
        item["rewards"] = sorted_rewards
        
    
    args.save_path = f"/home/linhaojia/liutao/train_model/prepare_rank_data/ranked_data_10s.json"
    with open(args.save_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def main(args):
    device = torch.device(f"cuda:{args.gpu_id}")
    
    # load data
    rank_dataset = Custome_Dataset(args.json_path)
    
    # load model
    if args.benchmark == 'ImageReward':
        eval_model = ImageReward_load(args.rm_path,device = device)
    elif args.benchmark == 'CLIP':
        eval_model = CLIPReward_load(weight = 'ViT-L/14', device=device, pretrained=True)

    eval_model.to(device)
    
    new_data = []
    with torch.no_grad():
        for i, (id, text, generations) in tqdm(enumerate(rank_dataset), total = len(rank_dataset)):
            print(id,text,generations)
            rewards = eval_model.score(text, generations)
            rewards = rewards.squeeze(1)
            rewards = rewards.detach().cpu().numpy().tolist()

            item = {}
            item['id'] = id
            item['text'] = text
            item['generations'] = generations
            item['rewards'] = rewards
            new_data.append(item)
            

    

    with open(args.save_path, 'w', encoding='utf-8') as json_file:
        json.dump(new_data, json_file, ensure_ascii=False, indent=4)

    # print(model_list)
    # print(win_num)


            
            
            
        
        


if __name__ == "__main__":
    args = parse_args()
    args.save_path = f"/home/linhaojia/liutao/train_model/prepare_rank_data/ranked_data_10.json"
    # main(args)
    sorted_by_rewards(args.save_path)

"""
python 2_eval_with_metric.py --gpu_id 0 --benchmark "ImageReward" 

['/data/liutao/images_ckpt/refl', '/data/liutao/images_ckpt/refl_4gpu']
* [42, 57]


['SD1.4', 'refl', 'rank_offline_3', 'rank_offline_4', 'rank_offline_7']
* [16, 30, 12, 14, 27]

['SD1.4', 'refl']
* [45, 54]

['SD1.4', 'rank_offline_3']
* [42, 57]

['SD1.4', 'rank_offline_4']
[44, 55]

['SD1.4', 'rank_offline_7']
* [42, 57]


python 2_eval_with_metric.py --gpu_id 0 --benchmark "CLIP"
['SD1.4', 'refl']
* [57, 42]

['SD1.4', 'rank_offline_7']
*  [51, 48]

['SD1.4', 'refl', 'S2C_rank', 'rank_offline_3', 'rank_offline_4', 'rank_offline_7']
* [17, 19, 11, 15, 17, 20]

['/data/liutao/images_ckpt/refl', '/data/liutao/images_ckpt/S2C_rank']
* [46, 53]

"""