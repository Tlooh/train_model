import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
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
import pdb



""" ===================== prepare components ========================="""

# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. schedule
scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)


"""===================== option parameters ========================="""
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument(
        "--json_path",
        default="/home/linhaojia/liutao/train_model/test/eval_json_99.json",
        type=str,
        help="Json path to Eval Model",
    )
    parser.add_argument(
        "--checkpoints", "-ckpt",
        default="SD1.4, refl",
        type=str,
        help="SD1.4, refl ……, splitted with comma(,) if there are multiple benchmarks.",
    )
    parser.add_argument(
        "--img_dir",
        default="/data/liutao/images_ckpt",
        type=str,
        help="Path to the save imgs generated per model",
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible Gen.")

    args = parser.parse_args()
    args.filename = os.path.basename(__file__)

    return args


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 {folder_path} 创建成功")
    else:
        print(f"文件夹 {folder_path} 已经存在") 


def gen_imgs(g, prompts, unet, device):
    batch_size = len(prompts)

    # 1. get input_ids
    text_inputs = tokenizer(
        prompts, 
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    guidance_scale = 7.5
    do_classifier_free_guidance = guidance_scale > 1.0
    text_input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask.to(device)

    # 2.get prompt embedding
    prompt_embeds = text_encoder(
                text_input_ids.to(device),
                # attention_mask=None,
            )
    prompt_embeds = prompt_embeds[0]

    # do_classifier_free_guidance
    uncond_tokens = [""] * batch_size
    max_length = prompt_embeds.shape[1]
    uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
    
    attention_mask = uncond_input.attention_mask.to(device)
    negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                # attention_mask=None,
            )
    negative_prompt_embeds = negative_prompt_embeds[0]

    # to avoid doing two forward passes
    # [8, 77, 768]
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 4. Prepare timesteps
    scheduler.set_timesteps(50, device=device)
    timesteps = scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = unet.config.in_channels
    shape = (batch_size, num_channels_latents, 64, 64)
    latents = torch.randn(shape, generator = g, device = device)
    
    for i, t in tqdm(enumerate(timesteps), total = 50):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
        
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy() # (bsz, 512, 512, 3)
    images = (image * 255).round().astype("uint8")

    pil_images = [Image.fromarray(image) for image in images]

    return pil_images





def main(args, unet):
    
    device = torch.device(f"cuda:{args.gpu_id}")
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # load data
    gen_dataset = Gen_Dataset(args.json_path)
    gen_loader = DataLoader(gen_dataset, batch_size=args.batch_size)

    for i, batch_data in tqdm(enumerate(gen_loader), total = len(gen_loader)):
        batch_ids, batch_texts, batch_entity = batch_data['idx'], batch_data['text'], batch_data['entity']
        
        # fixed seed
        g = torch.Generator(device=device).manual_seed(args.seed)
        # generate images
        pil_images = gen_imgs(g, batch_texts, unet, device)

        # save imgs
        for id, entity, img in zip(batch_ids, batch_entity, pil_images):
            img_name = f"{id}.png"
            img_path =  os.path.join(args.img_save_dir, img_name)
            img.save(img_path)


    print(f"{args.ckpt_name} generats imgs finished!")





if __name__ == "__main__":
    args = parse_args()

    ckpt_list = args.checkpoints.split(",")
    print(ckpt_list)

    for ckpt in ckpt_list:
        if ckpt == 'SD1.4':
            unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float32)
        else:
            weight_dir = "/data/liutao/checkpoints/SD_1-4"
            unet = UNet2DConditionModel.from_pretrained(f"{weight_dir}/{ckpt}/checkpoint-{args.steps}/unet", torch_dtype=torch.float32)
        
        args.ckpt_name = ckpt
        args.img_save_dir = os.path.join(args.img_dir, args.ckpt_name)
        create_folder_if_not_exists(args.img_save_dir)
        
        main(args, unet)


"""
python 1_gen_imgs.py --gpu_id 7 --checkpoints SD1.4,refl,rank_offline,rank_offline_gpu4
"""