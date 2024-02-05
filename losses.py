"""
Loss functions
* refl_loss: 原始 ImageReward 论文的 loss
"""

import torch
import torch.nn.functional as F

# ReFL loss
def refl_loss(rewards):
    loss = F.relu(-rewards+2)
    loss = loss.mean()

    return loss

# S2C loss
def rank_loss(rewards):
    # rewards [bsz, 1] -> [bsz/2, 2]
    # view 第一行是 simple, 第二行是 complex
    
    rewards = rewards.view(-1, 2)
    
    reward_diff = rewards[1, :] - rewards[0, :]
    
    loss = -torch.log(torch.sigmoid(reward_diff)).mean()
    
    return loss


# offline loss
def rank_offline_loss(rewards):
    # rewards: [num_rank, bsz]
    num_rank = rewards.size(0)
    K = num_rank * (num_rank - 1) / 2 if num_rank > 1 else  1
    loss_offline = torch.zeros(rewards.shape[-1], dtype=torch.float16).to(rewards.device)
    loss_offline.requires_grad = True
    for i in range(num_rank):
        for j in range(i+1, num_rank):
            loss_offline =loss_offline + rewards[i,:] - rewards[j,:]
            # print(f"i = {i}, j = {j} loss = {loss_offline}")
    
    loss_offline = loss_offline / K

    loss_offline = -torch.log(torch.sigmoid(loss_offline)).mean()

    # print(loss_offline)
    return loss_offline


# offline loss
def rank_offline_loss2(rewards):
    # rewards: [num_rank, bsz]
    loss_offline = rewards
    print(loss_offline)
    return loss_offline     
    


