import torch
import torch.nn as nn
import torch.nn.functional as F


class BPB(nn.Module):
    # Boundary Preserving Block (BPB) according to the paper, Structure Boundary Preserving Segmentation for Medical Image with Ambiguous Boundary

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.num_of_dilated_conv = 5
        remainder = out_channels % self.num_of_dilated_conv
        self.dilated_conv1 = nn.Conv3d(in_channels, out_channels // self.num_of_dilated_conv, kernel_size=1, dilation=1)
        self.dilated_conv2 = nn.Conv3d(in_channels, out_channels // self.num_of_dilated_conv, kernel_size=3, dilation=1)
        self.dilated_conv3 = nn.Conv3d(in_channels, out_channels // self.num_of_dilated_conv, kernel_size=3, dilation=2)
        self.dilated_conv4 = nn.Conv3d(in_channels, out_channels // self.num_of_dilated_conv, kernel_size=3, dilation=4)
        self.dilated_conv5 = nn.Conv3d(in_channels, out_channels // self.num_of_dilated_conv + remainder, kernel_size=3, dilation=6)

    def forward(self, x):
        # Boundary Point Map Generator
        d1 = self.dilated_conv1(x)
        d2 = self.dilated_conv2(x)
        d3 = self.dilated_conv3(x)
        d4 = self.dilated_conv4(x)
        d5 = self.dilated_conv5(x)

        # print(x.shape)

        # Zero pad to enable concat
        pd2 = (1, 1, 1, 1, 1, 1)
        d2 = F.pad(d2, pd2, "constant", 0)

        pd3 = (2, 2, 2, 2, 2, 2)
        d3 = F.pad(d3, pd3, "constant", 0)

        pd4 = (4, 4, 4, 4, 4, 4)
        d4 = F.pad(d4, pd4, "constant", 0)

        pd5 = (6, 6, 6, 6, 6, 6)
        d5 = F.pad(d5, pd5, "constant", 0)

        # Channel Wise Concatenation
        sigmoid_map = torch.sigmoid(torch.cat((d1, d2, d3, d4, d5), dim=1))

        # Multiplication and Addition
        # print((sigmoid_map * x + x).shape)
        return sigmoid_map * x + x


class BPB_4channels(nn.Module):
    # Boundary Preserving Block (BPB) according to the paper, Structure Boundary Preserving Segmentation for Medical Image with Ambiguous Boundary

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.num_of_dilated_conv = 4
        self.dilated_conv1 = nn.Conv3d(in_channels, out_channels // self.num_of_dilated_conv, kernel_size=1, dilation=1)
        self.dilated_conv2 = nn.Conv3d(in_channels, out_channels // self.num_of_dilated_conv, kernel_size=3, dilation=1)
        self.dilated_conv3 = nn.Conv3d(in_channels, out_channels // self.num_of_dilated_conv, kernel_size=3, dilation=2)
        self.dilated_conv4 = nn.Conv3d(in_channels, out_channels // self.num_of_dilated_conv, kernel_size=3, dilation=4)
        #self.dilated_conv5 = nn.Conv3d(in_channels, out_channels // self.num_of_dilated_conv, kernel_size=3, dilation=6)

    def forward(self, x):
        # Boundary Point Map Generator
        d1 = self.dilated_conv1(x)
        d2 = self.dilated_conv2(x)
        d3 = self.dilated_conv3(x)
        d4 = self.dilated_conv4(x)
        #d5 = self.dilated_conv5(x)

        # Zero pad to enable concat
        pd2 = (1, 1, 1, 1, 1, 1)
        d2 = F.pad(d2, pd2, "constant", 0)

        pd3 = (2, 2, 2, 2, 2, 2)
        d3 = F.pad(d3, pd3, "constant", 0)

        pd4 = (4, 4, 4, 4, 4, 4)
        d4 = F.pad(d4, pd4, "constant", 0)

        # pd5 = (6, 6, 6, 6, 6, 6)
        # d5 = F.pad(d5, pd5, "constant", 0)

        # Channel Wise Concatenation
        sigmoid_map = torch.sigmoid(torch.cat((d1, d2, d3, d4), dim=1))

        # Multiplication and Addition
        return sigmoid_map * x + x
