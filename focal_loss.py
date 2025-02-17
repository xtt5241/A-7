# focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    多标签二进制FocalLoss (with logits).
    默认 alpha=0.75, gamma=1.5, 若仍有全预测1现象可继续调低 gamma或alpha.
    """
    def __init__(self, alpha=0.75, gamma=1.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (batch, n_class), 未经过sigmoid
        targets: (batch, n_class), 0/1
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)

        # focal factor
        pt = torch.where(targets == 1, p, 1 - p)
        focal_factor = self.alpha * (1.0 - pt).pow(self.gamma)

        loss = focal_factor * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
