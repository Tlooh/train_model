import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from .BLIP.blip_pretrain import BLIP_Pretrain
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


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


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
    def forward(self, input):
        return self.layers(input)
    

class BLIPReward(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        self.blip = BLIP_Pretrain(image_size=224, vit='large')
        self.preprocess = _transform(224)
        self.mlp = MLP(768)

        # /data/liutao/checkpoints/BLIPReward/bs32_lr=5e-06_nce_s2c/best_acc=0.9236350576082866.pt
        self.mean = -0.0002
        self.std =  0.0019409654621058104
        
    def score_gard(self, prompt_ids, prompt_attention_mask, image):

        image_embeds = self.blip.visual_encoder(image)
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(prompt_ids,
                                                    attention_mask = prompt_attention_mask,
                                                    encoder_hidden_states = image_embeds,
                                                    encoder_attention_mask = image_atts,
                                                    return_dict = True,
                                                )
        
        txt_features = text_output.last_hidden_state[:,0,:] # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        
        return rewards
    
    def score(self, prompts, images):
        """
        prompts : text list
        images: path list
        """

        # text encode
        text_input = self.blip.tokenizer(prompts, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)

        # image encode
        if isinstance(images[0], str):
            pil_images = [Image.open(img) for img in images]
        images = [self.preprocess(img) for img in pil_images]

        images = torch.stack(images).to(self.device) # [num_ckpt, 3, 224, 224]
        image_embeds = self.blip.visual_encoder(images)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(text_input.input_ids,
                                                attention_mask = text_input.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,
                                            )
        txt_features = text_output.last_hidden_state[:,0,:].float() # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        
        return rewards

    


        
