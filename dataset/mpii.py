import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import cv2
from utils.gaussian import generate_gaussian
import matplotlib.pyplot as plt
from dataset.transform import train_transform
class MPIIDataset(Dataset):
    def __init__(self,path_images,keypoints_df,size=None,transforms=None,output_size=64):
        super(MPIIDataset, self).__init__()
        self.size=size
        self.path_images=path_images
        self.keypoints=keypoints_df
        self.transforms=transforms
        self.output_size=output_size

    def getimage (self,path):
        img = plt.imread(path)
        #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def getcoords(self,idx):
        columns_points=self.keypoints.columns[2:34]
        p=0
        coords={'r_ankle':[-1,-1], # 1
                'r_knee':[-1,-1],  # 2
                'r_hip':[-1,-1],   # 3
                'l_hip':[-1,-1],   # 4
                'l_knee':[-1,-1],  # 5
                'l_ankle':[-1,-1], # 6
                'pelvis':[-1,-1],  # 7
                'thorax':[-1,-1],  # 8
                'upperneck':[-1,-1], # 9
                'head_top':[-1,-1], # 10
                'r_wrist':[-1,-1], # 11
                'r_elbow':[-1,-1], # 12
                'r_shoulder':[-1,-1], # 13
                'l_shoulder':[-1,-1], # 14
                'l_elbow':[-1,-1], # 15
                'l_wrist':[-1,-1]} # 16
        for key in coords.keys():
            coords[key][0]=self.keypoints.iloc[idx][columns_points[p]]
            coords[key][1]=self.keypoints.iloc[idx][columns_points[p+1]]
            p+=2
        return coords
    def getscale(self,idx):
        return self.keypoints.iloc[idx]['Scale']
    def __getitem__(self, item):
        img_name=self.keypoints.iloc[item]['NAME']
        img_path=self.path_images+img_name
        image=self.getimage(img_path)
        coords=self.getcoords(item)
        scale=self.getscale(item)
        n_Joints=len(coords.keys())
        out=torch.zeros(n_Joints,self.output_size,self.output_size)


        keypoints=[(coords[key][0],coords[key][1]) for key in coords.keys()]
        if self.transforms:
            transformed=train_transform(image,keypoints,self.size,self.size)
            image_trans=transformed['image']
            coords_trans=transformed['keypoints']
            new_points=[]
            for coords in coords_trans:
                new_points.append((coords[0]/4,coords[1]/4))
            for i,coord in enumerate (new_points):
                out[i]=generate_gaussian(torch.zeros([64,64]),coord[0],coord[1])
        return {'images':image_trans,
                'keypoints':torch.tensor(coords_trans),
                'targets':out}
    def __len__(self):
        return len(self.keypoints)