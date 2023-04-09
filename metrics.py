import numpy as np
import torch
from medpy.metric.binary import jc, dc
import torch.nn as nn


class IOU(nn.Module):
    def __init__(self):
        super(IOU, self).__init__()
        self.mean_iou = 0
        self.count = 0

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1).cpu().detach().numpy()
        target = target.contiguous().view(target.shape[0], -1).cpu().detach().numpy()

        self.mean_iou += jc(predict, target)
        self.count += 1

        return self.mean_iou / self.count


class DICE(nn.Module):
    def __init__(self):
        super(DICE, self).__init__()
        self.mean_dice = 0
        self.count = 0

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1).cpu().detach().numpy()
        target = target.contiguous().view(target.shape[0], -1).cpu().detach().numpy()

        self.mean_dice += dc(predict, target)
        self.count += 1

        return self.mean_dice / self.count
