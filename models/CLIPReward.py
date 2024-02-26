import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

from PIL import Image
from tqdm import tqdm
import clip
import json
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from utils import *
from .CLIP.model import clip_pretrain


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


# ViT-L/14 
kwargs = {'embed_dim': 768, 'image_resolution': 224, 'vision_layers': 24, 'vision_width': 1024, 'vision_patch_size': 14, 'context_length': 77, 'vocab_size': 49408, 'transformer_width': 768, 'transformer_heads': 12, 'transformer_layers': 12}


"""
clip_name: 
* 若为 ViT-L/14 ,则加载预训练模型
* 若为权重 /path/to/bs32_lr=5e-06_sig1.pt, 则是加载自己训练的权重

"""

class CLIPReward(nn.Module):
    def __init__(self, clip_name = None, device = 'cpu'):
        super().__init__()
        self.device = device
        self.clip_model = clip_pretrain(pretrained=clip_name, **kwargs)
        self.preprocess = _transform(224)

        # TODO: 计算 mean 、 std
    
    def score_gard(self, prompt_ids, prompt_attention_mask, image):
        # TODO: 计算 mean 、 std
        pass

    def score(self, prompts, images):
        """
        prompts : text list
        images: path list
        """
        # text tokenizer
        text_tokens = clip.tokenize(prompts).to(self.device)

        # images：str -> pil ->tensor
        images = [self.preprocess(Image.open(img)) for img in images]
        images = torch.stack(images).to(self.device) #[bsz, 3, 224, 224]
        
        image_embeds = self.clip_model.encode_image(images)
        text_embeds = self.clip_model.encode_text(text_tokens)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
        
        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        rewards = logit_scale * image_embeds @ text_embeds.t()
        
        return rewards


        

    


def CLIPReward_load(weight, device, pretrained=False):
    if pretrained:
        model = CLIPReward(clip_name=weight, device=device).to(device)
    else:
        state_dict = torch.load(weight, map_location='cpu')
        model = CLIPReward(clip_name=None, device=device).to(device)
        msg = model.load_state_dict(state_dict, strict=False)
    print("checkpoint loaded")
    model.eval()
    return model
    
