import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
import argparse
import sys
sys.path.append('/home/linhaojia/liutao/train_model')
import json
import torch 

from utils import *




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
        "--steps",
        default="100",
        type=str,
        help="……/SD_1-4/rank_offline/checkpoint-100",
    )
    parser.add_argument(
        "--gpu_id",
        default=7,
        type=str,
        help="GPU ID(s) to use for CUDA.",
    )
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible Gen.")

    args = parser.parse_args()
    args.filename = os.path.basename(__file__)

    return args



if __name__ == "__main__":
    args = parse_args()

    ckpt_list = [
        "SD1.4",
        "/data/liutao/checkpoints/SD_1-4/refl"
    ]

    print("Generate Images with CKPT_LIST:")
    main_gen(args, ckpt_list)

    model_list = [
        "/data/liutao/images_ckpt/SD1.4",
        "/data/liutao/images_ckpt/T2I"
    ]

    do_eval_with_metric(args, model_list)


"""
python gen_eval.py --gpu_id 0 
"""