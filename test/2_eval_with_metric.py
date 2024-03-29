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
    model_list = ['SD1.4', 'refl', 'S2C_rank','rank_offline_3', 'rank_offline_4', 'rank_offline_7']
    model_list = ['SD1.4', 'refl', 'rank_offline_3', 'rank_offline_4', 'rank_offline_7']
    # model_list = ['refl','refl_4gpu']
    # model_list = ['refl','S2C_CustomBLIP']
    # model_list = ['SD1.4', 'rank_offline_3', 'rank_offline_4', 'rank_offline_7']
    # model_list = ['SD1.4', 'rank_offline_3', 'rank_offline_4', 'rank_offline_7']
    model_list = ['refl', 'refl_myreward']
    # model_list = ['SD1.4','refl_myreward']
    model_list =[os.path.join(args.img_dir, model_name) for model_name in model_list]

    # load data
    eval_dataset = Eval_Dataset(args.json_path, model_list=model_list)
    
    # load model
    if args.benchmark == 'ImageReward':
        eval_model = ImageReward_load(args.rm_path,device = device)
    elif args.benchmark == 'CLIP':
        eval_model = CLIPReward_load(weight = 'ViT-L/14', device=device, pretrained=True)

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

    print(model_list)
    print(win_num)


            
            
            
        
        


if __name__ == "__main__":
    args = parse_args()

    main(args)

"""
python 2_eval_with_metric.py --gpu_id 1 --benchmark "ImageReward" 

['/data/liutao/images_ckpt/refl', '/data/liutao/images_ckpt/refl_4gpu']
* [42, 57]


['SD1.4', 'refl', 'rank_offline_3', 'rank_offline_4', 'rank_offline_7']
* [16, 30, 12, 14, 27]

['SD1.4', 'refl']
* [45, 54]

['SD1.4', 'T2I']
* [46, 53]

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

['/data/liutao/images_ckpt/SD1.4', '/data/liutao/images_ckpt/refl_myreward']
[45, 54]


"""

"""
1624

python 2_eval_with_metric.py --gpu_id 6 --benchmark "ImageReward"  --json_path "/home/linhaojia/liutao/train_model/test/eval_1624.json" --img_dir "/data/liutao/images_1624"

model_list = ['SD1.4', 'refl']
[695, 929]

['/data/liutao/images_1624/SD1.4', '/data/liutao/images_1624/T2I']
[744, 880]

['/data/liutao/images_1624/refl', '/data/liutao/images_1624/T2I']
[817, 807]

['/data/liutao/images_1624/SD1.4', '/data/liutao/images_1624/refl_myreward']
[686, 938]

['/data/liutao/images_1624/refl', '/data/liutao/images_1624/refl_myreward']
[839, 785]
"""