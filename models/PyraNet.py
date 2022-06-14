import torch
import torch.nn as nn
from models.PRM import PRM
from models.Hourglass import Hourglass

class PRM_Pool (nn.Module):
    def __init__(self,in_features,out_features,c):
        super(PRM_Pool, self).__init__()
        self.prm=PRM(in_features,out_features,c)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
    def forward (self,x):
        return self.pool(self.prm(x))
class Pyranet (nn.Module):
    # пока пусть стак будет равен 1
    def __init__(self,n_features,nPRM,c,n_Joints,n_Hourglass=1):
        super(Pyranet, self).__init__()

        self.n_features=n_features
        self.n_Hourglass=n_Hourglass
        self.n_Joints=n_Joints
        self.nPRM=nPRM
        self.c=c

        # -----Финальная модель------
        self.conv=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=1,stride=2)
        self.prm_pool=PRM_Pool(64,64,self.c)
        self.prm1=PRM(64,self.n_features,self.c)
        self.relu=nn.ReLU()

        self.hourglass=Hourglass(2,self.nPRM,self.c,self.n_features)
        self.prms=[PRM(n_features,n_features,self.c) for i in range(self.nPRM)]

        self.lin = nn.Sequential(nn.Conv2d(self.n_features, self.n_features, bias=True, kernel_size=1, stride=1),
                            nn.BatchNorm2d(self.n_features),
                            self.relu)

        self.tmpout=nn.Conv2d(self.n_features, self.n_Joints, bias = True, kernel_size = 1, stride = 1)

    def forward (self,x):
        #256x256
        x=self.conv(x) #->128x128
        x=self.prm_pool(x) #->64x64
        x=self.prm1(x) #->64x64

        hg=self.hourglass(x)
        li=hg
        for prm in self.prms:
            li=prm(li)

        li = self.lin(li)
        tmpOut = self.tmpout(li)

        return tmpOut



