import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
  def __init__(self, device, alpha=None, gamma=2, smooth=1e-5, size_average=True):
    super(FocalLoss, self).__init__()
    self.device = device
    self.alpha = alpha
    self.gamma = gamma
    self.smooth = smooth
    self.size_average = size_average

    if self.smooth is not None:
      if self.smooth < 0 or self.smooth > 1.0:
        raise ValueError('Smooth value should be in [0,1]')

  def forward(self, logit, target, alpha=None):
    N = logit.size(0)
    C = logit.size(1)

    if logit.dim() > 2:
      # N, C, d1, d2, ..., dn --> N * d1 * d2 * ... * dn, C
      logit = logit.view(N, C, -1)
      logit = logit.permute(0, 2, 1).contiguous()
      logit = logit.view(-1, C)

    # N, d1, d2, ..., dn --> N * d1 * d2 * ... * dn, 1
    target = torch.squeeze(target, 1)
    target = target.view(-1, 1)

    idx = target.cpu().long()
    one_hot_key = torch.FloatTensor(target.size(0), C).zero_()
    one_hot_key = one_hot_key.scatter_(1, idx, 1)
    one_hot_key = torch.clamp(one_hot_key,
                              self.smooth / (C - 1),
                              1.0 - self.smooth)
    one_hot_key = one_hot_key.to(self.device)

    pt = (one_hot_key * logit).sum(1) + self.smooth
    logpt = pt.log()

    if alpha is None:
      alpha = self.alpha.to(self.device)
    alpha = alpha[idx].squeeze()
    loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt
    loss = loss.view(N, -1)

    if self.size_average:
      loss = loss.mean(1)
    else:
      loss = loss.sum(1)

    return loss

