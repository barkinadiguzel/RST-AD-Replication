import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, student_feat, teacher_feat):
        attn = self.conv3x3(teacher_feat)
        attn = self.bn(attn)
        attn = self.relu(attn)
        attn = self.conv1x1(attn)
        attn = self.sigmoid(attn)
        out = student_feat * attn
        return out
