""" Full assembly of the parts to form the complete network """

from unet_parts import *
from BPB import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=True, use_BPB=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear
        self.use_BPB = use_BPB

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)

        if use_BPB:
            self.bpb1 = BPB(256, 256)

        self.down3 = Down(256, 512)
        factor = 2 if trilinear else 1
        self.down4 = Down(512, 1024 // factor)

        if use_BPB:
            self.bpb2 = BPB_4channels(1024 // factor, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, trilinear)
        self.up2 = Up(512, 256 // factor, trilinear)

        if use_BPB:
            self.bpb3 = BPB(256 // factor, 256 // factor)

        self.up3 = Up(256, 128 // factor, trilinear)
        self.up4 = Up(128, 64, trilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        if self.use_BPB:
            b1 = self.bpb1(x3)

            x4 = self.down3(b1)
            x5 = self.down4(x4)
            b2 = self.bpb2(x5)

            x = self.up1(b2, x4)
            x = self.up2(x, x3)
            b3 = self.bpb3(x)

            x = self.up3(b3, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
        else:
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)

        return logits