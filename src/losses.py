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

