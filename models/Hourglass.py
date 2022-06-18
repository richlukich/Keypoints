import torch
import torch.nn as nn
from Keypoints.models.PRM import PRM

class DownSampling(nn.Module):
    def __init__(self):
        super(DownSampling, self).__init__()
        self.downsample=nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,x):
        return self.downsample(x)

class UpSampling (nn.Module):
    def __init__(self,):
        super(UpSampling, self).__init__()
        self.upsample=nn.Upsample(scale_factor=2)
    def forward (self,x):
        return self.upsample(x)

class Hourglass (nn.Module):
    def __init__(self,n,nPRM,c,n_features):
        super(Hourglass, self).__init__()
        self.n=n
        self.n_features=n_features
        self.nPRM=nPRM
        self.c=c

        self.up1=nn.ModuleList([PRM(n_features,n_features,c) for i in range (self.nPRM)])
        self.pool1=DownSampling()
        self.low1=nn.ModuleList([PRM(n_features,n_features,c) for i in range (self.nPRM)])
        #Рекурсия часов :)
        if self.n > 1:
            self.low2 = Hourglass(n-1,nPRM,c,n_features)
        else:
            self.low2 = nn.ModuleList([PRM(n_features,n_features,c) for i in range (self.nPRM)])

        self.low3=nn.ModuleList([PRM(n_features,n_features,c) for i in range (self.nPRM)])
        self.up2=UpSampling()



    def forward (self,x):
        up1=x
        for prm in self.up1:
            up1=prm(up1)
        low1=self.pool1(up1)

        for prm in self.low1:
            low1=prm(low1)


        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for prm in self.low2:
                low2 = prm(low2)

        low3=low2
        for prm in self.low3:
            low3=prm(low3)
        up2=self.up2(low3)

        return up1+up2







