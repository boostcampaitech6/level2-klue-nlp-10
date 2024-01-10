import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean') -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.CrossEntropyLoss(reduction='none')
        ce_loss = ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt)**self.gamma*ce_loss
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()