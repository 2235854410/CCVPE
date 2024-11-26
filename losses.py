import torch
import torch.nn as nn

def infoNCELoss(scores, labels, temperature=0.1):
    """
    Contrastive loss over matching score. Adapted from https://arxiv.org/pdf/2004.11362.pdf Eq.2
    We extraly weigh the positive samples using the ground truth likelihood on those positions
    
    loss = - 1/sum(weights) * sum(inner_element*weights)
    inner_element = log( exp(score_pos/temperature) / sum(exp(score/temperature)) )
    """
    
    exp_scores = torch.exp(scores / temperature)
    bool_mask = labels>1e-2 # elements with a likelihood > 1e-2 are considered as positive samples in contrastive learning    
    
    denominator = torch.sum(exp_scores, dim=1, keepdim=True)
    inner_element = torch.log(torch.masked_select(exp_scores/denominator, bool_mask))
    loss = -torch.sum(inner_element*torch.masked_select(labels, bool_mask)) / torch.sum(torch.masked_select(labels, bool_mask))
    
    return loss


def cross_entropy_loss(logits, labels):
    return -torch.sum(labels * nn.LogSoftmax(dim=1)(logits)) / logits.size()[0]

        
    
def orientation_loss(ori, gt_orientation, gt):    
    return torch.sum(torch.sum(torch.square(gt_orientation-ori), dim=1, keepdim=True) * gt) / ori.size()[0]

def contrastive_loss(score_map, labels, temperature=0.1):
    B, _, _, H, W = score_map.shape

    # 将 score_map 和 labels 展开，以计算 Softmax 概率
    scores = score_map.view(B, B, -1) / temperature  # (B, B, H*W)

    bool_mask = labels.view(B, -1) > 1e-2  # 展开后的 bool_mask，(B, H*W)

    # 计算 exp(scores) 并进行归一化
    exp_scores = torch.exp(scores)
    denom = torch.sum(exp_scores, dim=-1, keepdim=True)  # (B, B, 1)
    denom = torch.sum(denom, dim=1) # (B, 1)

    # 计算每个正样本对的 softmax 概率，仅包括 gt 点
    # prob_pos = (exp_scores.diagonal(dim1=0, dim2=1) / denom.squeeze(-1))  # (B, H*W)
    exp_pos_samples = exp_scores[torch.arange(B), torch.arange(B)] #[4, 64]
    gt_prob = exp_pos_samples * bool_mask
    prob_pos = (exp_pos_samples / denom)  # (B, H*W)

    # 对正样本概率取对数，并计算带权重的损失
    log_prob_pos = torch.log(prob_pos + 1e-10)  # 加一个小数防止 log(0)
    loss_pos = -torch.sum(log_prob_pos.to(labels.device) * bool_mask.float()) / torch.sum(bool_mask.float())  # 只对 gt 点求平均

    return loss_pos


def triplet_loss(score_maps, labels):
    matching_losses = []
    for i in range(len(score_maps)):
        B, _, _, H, W = score_maps[i].shape

        # 将 score_map 和 labels 展开，以计算 Softmax 概率
        scores = score_maps[i].view(B, B, -1)  # (B, B, H*W)

        max_scores = torch.max(scores, dim=-1)[0] # (B, B)
        pos = torch.diagonal(max_scores) # (B)
        delta = max_scores - pos.reshape(-1, 1) # big delta means negative localization have high confidence
        loss = torch.sum(torch.log(1 + torch.exp(delta * 10))) / (B * (B - 1))

        matching_losses.append(loss)

    return torch.mean(torch.stack(matching_losses, dim=0))

def contrastive_loss_only_pos(score_map, labels, temperature=0.1):
    B, _, H, W = score_map.shape

    # 将 score_map 和 labels 展开，以计算 Softmax 概率
    scores = score_map.view(B, -1) / temperature  # (B, H*W)
    bool_mask = labels.view(B, -1) > 1e-2  # 展开后的 bool_mask，(B, H*W)

    # 计算 exp(scores) 并进行归一化
    exp_scores = torch.exp(scores)

    # 计算每个正样本对的 softmax 概率，仅包括 gt 点
    # prob_pos = (exp_scores.diagonal(dim1=0, dim2=1) / denom.squeeze(-1))  # (B, H*W)
    # exp_pos_samples = exp_scores[torch.arange(B), torch.arange(B)] #[4, 64]
    denom_pos = torch.sum(exp_scores, dim=1, keepdim=True)
    prob_pos = (exp_scores / denom_pos)  # (B, H*W)

    # 对正样本概率取对数，并计算带权重的损失
    log_prob_pos = torch.log(prob_pos + 1e-10)  # 加一个小数防止 log(0)
    loss_pos = -torch.sum(log_prob_pos.to(labels.device) * bool_mask.float()) / torch.sum(bool_mask.float())  # 只对 gt 点求平均

    return loss_pos
