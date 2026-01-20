from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F


def extract_feature_mask(input_mask, T_feat):
    """
    input_mask: [T_in=160], bool or {0,1} indicating which input frames are kept.
    T_feat: feature time dimension, e.g. 10.

    Returns:
        feature_mask: [T_feat], float {0,1}
                       1 = unmasked / kept
                       0 = masked
    """
    T_in = input_mask.shape[0]
    ratio = T_in // T_feat   # should be 16

    # reshape into chunks that correspond to receptive fields of feature steps
    x = input_mask.view(T_feat, ratio).float()

    # majority vote inside each chunk
    feature_mask = (x.mean(dim=1) > 0.5).float()

    return feature_mask   # length = T_feat


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """

    # def __init__(self, in_c, out_c, k, s=1, p=0):
    #     super(ConvBlock, self).__init__()
    #     self.conv = nn.Conv1d(in_c, out_c, k)
    #     self.bn = nn.BatchNorm1d(out_c)

    # def forward(self, x):
    #     return self.bn(self.conv(x))
    
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        # self.conv1 = ConvBlock(10, 64, 1)

        # self.conv1 = nn.Conv2d(10, 64, 1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(64, 10, 1, stride=1, padding=0)

        # self.conv1 = nn.Conv2d(1280, 3, 1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(3, 1280, 1, stride=1, padding=0)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))

        # self.conv1 = nn.Conv1d(160, 3, 1)
        # self.conv2 = nn.Conv1d(3, 160, 1)
        
        self.conv1 = nn.Conv1d(25, 3, 1)
        self.conv2 = nn.Conv1d(3, 25, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # self.conv1 = nn.Conv2d(25, 3, 1)
        # self.conv2 = nn.Conv2d(3, 25, 1)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))


    def get_attention(self, a):
        input_a = a

        a = a.mean(3)  # (B, N1, N2, T, T) -> (B, N1, N2, T)
        # a = a.transpose(1, 3)
        a = a.view(input_a.size(0), -1, input_a.size(-1))  # (B, N1*N2, T)
        # a = a.view(input_a.size(0), -1, input_a.size(-1), input_a.size(-1))  # (B, N1*N2, T, T)

        a = F.relu(self.conv1(a))
        a = self.conv2(a)
        
        # a = a.transpose(1, 3)
        a = a.view(input_a.size(0), input_a.size(1), input_a.size(2), input_a.size(-1))  # (B, N1, N2, T)
        a = a.unsqueeze(3)
        # a = a.view(input_a.size(0), input_a.size(1), input_a.size(2), input_a.size(-1), input_a.size(-1))  # (B, N1, N2, T, T)

        a = torch.mean(input_a * a, -1)
        # print(a[0])
        a = F.softmax(a / 0.025, dim=-1) + 1
        # a = F.sigmoid(a / 0.025) + 1

        # a = (a - a.mean(dim=-1, keepdim=True)) / (a.std(dim=-1, keepdim=True) + 1e-6)
        # print(a[0])
        # a = F.softmax(a, dim=-1) + 1

        # print(a[0].max(), a[0].min())

        return a

    def forward(self, f1, f2):
        b, n1, c, h, w = f1.size() # (1, 5, 128, 8, 160)
        n2 = f2.size(1) # (1, 5, 128, 8, 160)
        # print(f1.shape, f2.shape)

        # f1 = f1.view(b, n1, c, -1)
        # f2 = f2.view(b, n2, c, -1)  #(b, N2, c, h*w )

        # Stack frequency into channel dimension
        f1 = f1.view(b, n1, c * h, w)
        f2 = f2.view(b, n2, c * h, w)

        # f1 = f1.mean(-1) # over freq
        # f2 = f2.mean(-1)
        # f1 = f1.mean(-2) # over time shape: (b, N1, c, T) (1,5,128,160)
        # f2 = f2.mean(-2)

        # print(f1.shape, f2.shape)

        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)
        # print(f1_norm.shape, f2_norm.shape)

        f1n = f1_norm.transpose(2, 3).unsqueeze(2)   # [B, N1, 1, T, C]
        f2n = f2_norm.unsqueeze(1)                   # [B, 1, N2, C, T]
        # print(f1n.shape, f2n.shape)
        a1 = torch.matmul(f1n, f2n)                  # [B, N1, N2, T, T]
        a2 = a1.transpose(3, 4)                      # [B, N1, N2, T, T]

        a1 = self.get_attention(a1)                  # [B, N1, N2, T]
        a2 = self.get_attention(a2)                  # [B, N1, N2, T]

        # print(a1.shape, a2.shape)
        # print(f1.shape, f2.shape)

        f1 = f1.unsqueeze(2) * a1.unsqueeze(3)  # [B, N1, N2, C, T]
        f2 = f2.unsqueeze(1) * a2.unsqueeze(3)  # [B, N1, N2, C, T]
        # print(f1.shape, f2.shape)

        # f1 = f1.unsqueeze(-2)  # [B, N1, N2, C, T, 1]
        # f2 = f2.unsqueeze(-2)  # [B, N1, N2, C, T, 1]
        # print(f1.shape, f2.shape)

        f1 = f1.view(b, n1, n2, c, h, w)
        f2 = f2.view(b, n1, n2, c, h, w)

        return f1.transpose(1, 2), f2.transpose(1, 2), a1, a2