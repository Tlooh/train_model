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
import pdb



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument(
        "--json_path",
        default="/home/linhaojia/liutao/train_model/test/eval_json_99.json",
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
        "--result_dir",
        default="./benchmark_ImageReward",
        type=str,
        help="Path to the metric results directory",
    )
    parser.add_argument(
        "--img_dir",
        default="/data/liutao/images_ckpt",
        type=str,
        help="Path to the save imgs generated per model",
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



def main(args):
    device = torch.device(f"cuda:{args.gpu_id}")
    model_list = ['SD1.4', 'refl', 'rank_offline', 'rank_offline_gpu4']
    model_list =[os.path.join(args.img_dir, model_name) for model_name in model_list]

    # load data
    eval_dataset = Eval_Dataset(args.json_path, model_list=model_list)
    
    # load model
    if args.benchmark == 'ImageReward':
        eval_model = ImageReward_load(args.rm_path,device = device)

    eval_model.to(device)
    win_num = [0] * len(model_list)
    with torch.no_grad():
        for i, (id, text, imgs) in tqdm(enumerate(eval_dataset), total = len(eval_dataset)):
            # print(imgs)
            rewards = eval_model.score(text, imgs) # [num_ckpt, 1]
            rewards = rewards.squeeze(1)
            # print(rewards)
            max_Index = torch.argmax(rewards)
            win_num[max_Index] += 1
            
    print(win_num)


            
            
            
        
        


if __name__ == "__main__":
    args = parse_args()

    main(args)

"""
python 2_eval_with_metric.py --gpu_id 7 --benchmark "ImageReward" 
"""