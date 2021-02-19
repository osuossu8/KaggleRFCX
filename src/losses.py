import sys
sys.path.append("/root/workspace/KaggleRFCX")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


class PANNsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()

    def forward(self, input, target):
        input_ = input["clipwise_output"]
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)

        target = target.float()

        return self.bce(input_, target)


class ClassWeightedPANNsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss(reduction='none')

        self.weights = torch.tensor([0.04, 0.04, 0.02, 0.07, 0.04, 0.04,
                                     0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                                     0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                                     0.06, 0.02, 0.04, 0.07, 0.04, 0.04]).cuda()

    def forward(self, input, target):
        input_ = input["clipwise_output"]
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)

        target = target.float()

        loss = self.bce(input_, target)

        return (loss * self.weights).mean()


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = 2

    def forward(self, input, target):
        # input_ = input["clipwise_output"]
        input_ = input["logit"]
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)

        target = target.float()

        bce_loss = self.bce(input_, target)
        probas = torch.sigmoid(input_)
        loss = torch.where(target >= 0.5, (1. - probas)**self.gamma * bce_loss, probas**self.gamma * bce_loss)
        return loss.mean()


class FocalLoss5th(nn.Module):
    def __init__(self, stage=2, gamma=2.0, alpha=1.0):
        super().__init__()
        self.posi_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.nega_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.zero_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha
        self.stage = stage
        # self.zero_smoothing = 0.45

    def forward(self, input, target):
        # mask
        posi_mask = (target == 1).float()
        nega_mask = (target == -1).float()  # (n_batch, n_class)
        zero_mask = (target == 0).float()  # ambiguous label
      
        posi_y = torch.ones(input.shape).to('cuda')
        nega_y = torch.zeros(input.shape).to('cuda')
        zero_y = torch.full(input.shape, self.zero_smoothing_label).to('cuda')   # use smoothing label

        posi_loss = self.posi_loss(input, posi_y)
        nega_loss = self.nega_loss(input, nega_y)
        zero_loss = self.zero_loss(input, zero_y)
        
        probas = input.sigmoid()
        focal_pw = (1. - probas)**self.gamma
        focal_nw = probas**self.gamma
        posi_loss = (posi_loss * posi_mask * focal_pw).sum()
        nega_loss = (nega_loss * nega_mask).sum()
        zero_loss = (zero_loss * zero_mask).sum()  # stage2ではこれをlossに加えない
        
        if stage == 2:
            return posi_loss + nega_loss
        else:
            return posi_loss + nega_loss + zero_loss


def focal_loss(input, target, focus=2.0, raw=True):

    if raw:
        input = torch.sigmoid(input)

    eps = 1e-7

    prob_true = input * target + (1 - input) * (1 - target)
    prob_true = torch.clamp(prob_true, eps, 1-eps)
    modulating_factor = (1.0 - prob_true).pow(focus)

    return (-modulating_factor * prob_true.log()).mean()


def binary_cross_entropy(input, target, raw=True):
    if raw:
        input = torch.sigmoid(input)
    return torch.nn.functional.binary_cross_entropy(input, target)


def lsep_loss_stable(input, target, average=True):

    n = input.size(0)

    differences = input.unsqueeze(1) - input.unsqueeze(2)
    where_lower = (target.unsqueeze(1) < target.unsqueeze(2)).float()

    differences = differences.view(n, -1)
    where_lower = where_lower.view(n, -1)

    max_difference, index = torch.max(differences, dim=1, keepdim=True)
    differences = differences - max_difference
    exps = differences.exp() * where_lower

    lsep = max_difference + torch.log(torch.exp(-max_difference) + exps.sum(-1))

    if average:
        return lsep.mean()
    else:
        return lsep


def lsep_loss(input, target, average=True):

    differences = input.unsqueeze(1) - input.unsqueeze(2)
    where_different = (target.unsqueeze(1) < target.unsqueeze(2)).float()

    exps = differences.exp() * where_different
    lsep = torch.log(1 + exps.sum(2).sum(1))

    if average:
        return lsep.mean()
    else:
        return lsep


class LSEPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # input_ = input["clipwise_output"]
        input_ = input["logit"]
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)

        target = target.float()

        return lsep_loss(input_, target)


class PANNsWithFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = 2

    def forward(self, input, target):
        input_c = input["clipwise_output"]
        input_l = input["logit"]
        input_c = torch.where(torch.isnan(input_c),
                             torch.zeros_like(input_c),
                             input_c)
        input_c = torch.where(torch.isinf(input_c),
                             torch.zeros_like(input_c),
                             input_c)

        input_l = torch.where(torch.isnan(input_l),
                             torch.zeros_like(input_l),
                             input_l)
        input_l = torch.where(torch.isinf(input_l),
                             torch.zeros_like(input_l),
                             input_l)

        target = target.float()

        bce_with_logits_loss = self.bce_with_logits(input_l, target)
        probas = torch.sigmoid(input_l)
        loss = torch.where(target >= 0.5, (1. - probas)**self.gamma * bce_with_logits_loss, probas**self.gamma * bce_with_logits_loss)
        return (loss.mean() + self.bce(input_c, target))/2

