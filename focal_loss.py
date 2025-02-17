# focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    多标签二进制FocalLoss (with logits):
      alpha用于平衡正负例，gamma调节难易样本。常见设定: alpha=1, gamma=2.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (batch, n_class), 未经过sigmoid
        targets: (batch, n_class), 0/1
        """
        # BCE: -[ y * log(sigmoid(x)) + (1-y)*log(1-sigmoid(x)) ]
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # p = sigmoid(x)
        p = torch.sigmoid(logits)

        # focal: factor = alpha * (1-pt)^gamma
        #   对于正例(y=1): pt = p
        #   对于负例(y=0): pt = 1 - p
        pt = torch.where(targets == 1, p, 1 - p)
        focal_factor = self.alpha * (1.0 - pt).pow(self.gamma)

        loss = focal_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
