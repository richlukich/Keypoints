import torch.nn as nn
import torch
import torch.nn.functional as F

class BN_ReLU_1x1 (nn.Module):
    def __init__(self,in_channels,out_channels):
        super(BN_ReLU_1x1, self).__init__()

        self.bn=nn.BatchNorm2d(in_channels)
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=0)
        self.relu=nn.ReLU()

    def forward(self,x):
        x_bn=self.bn(x)
        x_conv=self.conv(x_bn)
        return self.relu(x_conv)

class BN_ReLU_3x3 (nn.Module):
    def __init__(self,in_channels,out_channels):
        super(BN_ReLU_3x3, self).__init__()

        self.bn=nn.BatchNorm2d(in_channels)
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1)
        self.relu=nn.ReLU()

    def forward (self,x):
        x_bn = self.bn(x)
        x_conv = self.conv(x_bn)
        return self.relu(x_conv)
class UpSampling (nn.Module):
    def __init__(self,scale):
        super(UpSampling, self).__init__()
        self.scale=scale
        self.sample=nn.Upsample(scale_factor=self.scale)
    def forward (self,x):
        return self.sample(x)
class DownSampling (nn.Module):
    def __init__(self,scale):
        super(DownSampling, self).__init__()
        self.scale=scale
    def forward(self,x):
        return F.interpolate(x,scale_factor=self.scale)

class Ratio (nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Ratio, self).__init__()
        self.convbn1=BN_ReLU_1x1(in_channels=in_channels,out_channels=out_channels//2)
        self.downsample=DownSampling(scale=0.5)
        self.convbn3=BN_ReLU_3x3(in_channels=out_channels//2,out_channels=out_channels)
        self.upsample=UpSampling(scale=2)

    def forward (self,x):
        x=self.convbn1(x)
        x=self.downsample(x)
        x=self.convbn3(x)
        x=self.upsample(x)
        return x

class PRM (nn.Module):
    def __init__(self,in_channels,out_channels,c):
        super(PRM, self).__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.c=c

        self.f0=nn.Sequential(BN_ReLU_1x1(in_channels=self.in_channels,out_channels=self.out_channels//2),
                          DownSampling(scale=0.5),
                         BN_ReLU_3x3(in_channels=self.out_channels//2,out_channels=self.out_channels),
                         UpSampling(scale=2),
                          BN_ReLU_1x1(in_channels=self.out_channels,out_channels=self.out_channels))

        #self.ratio=Ratio(in_channels=in_channels,out_channels=out_channels)
        self.ratio1=Ratio(in_channels=in_channels,out_channels=out_channels)
        self.ratio2=Ratio(in_channels=in_channels,out_channels=out_channels)
        self.ratio3=Ratio(in_channels=in_channels,out_channels=out_channels)

        #self.f1_fc=nn.ModuleList([self.ratio for i in range(c)])
        self.conv=BN_ReLU_1x1(in_channels=self.out_channels,out_channels=self.out_channels)

    def sum_f1_fc(self,x):
        x1=0
        for i in range (self.c):
            x1+=self.f1_fc[i](x)
        return x1
    def forward(self,x):
        f0=self.f0(x)
        #f1_fc=self.sum_f1_fc(x)
        ratio1=self.ratio1(x)
        ratio2=self.ratio2(x)
        ratio3=self.ratio3(x)
        f1_fc=ratio1+ratio2+ratio3
        g=self.conv(f1_fc)
        x1=f0+g
        return x1






