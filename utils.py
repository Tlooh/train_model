import torch
from models.ImageReward import ImageReward
from models.BLIPReward import BLIPReward



def ImageReward_load(weight_path, device):
    print('load checkpoint from %s'%weight_path)
    state_dict = torch.load(weight_path, map_location='cpu')

    model = ImageReward(device=device).to(device)
    msg = model.load_state_dict(state_dict, strict=False)
    print("checkpoint loaded")
    model.eval()

    return model


def BLIPReward_load(weight_path, device):
    print('load checkpoint from %s'%weight_path)
    state_dict = torch.load(weight_path, map_location='cpu')

    model = BLIPReward(device=device).to(device)
    msg = model.load_state_dict(state_dict, strict=False)
    print("checkpoint loaded")
    model.eval()

    return model
