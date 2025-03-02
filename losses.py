import torch
import torch.nn as nn
import torch.nn.functional as F
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


def cross_entropy(pred, target, s_temp=0.06, t_temp=0.06, eps=1e-8):
    b = pred.shape[0]

    # 使用 log_softmax 替代 softmax + log 操作
    pred_softmax = F.softmax(pred / s_temp, dim=1)
    target_softmax = F.softmax(target / t_temp, dim=1)

    # 添加极小值 eps 避免 log(0)
    loss = -torch.sum(target_softmax * torch.log(pred_softmax + eps), dim=1)
    return torch.mean(loss)

def orientation_loss(ori, gt_orientation, gt):    
    return torch.sum(torch.sum(torch.square(gt_orientation-ori), dim=1, keepdim=True) * gt) / ori.size()[0]
