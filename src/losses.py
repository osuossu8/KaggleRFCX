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
