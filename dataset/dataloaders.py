from dataset.mpii import MPIIDataset
import torch
from torch.utils.data import DataLoader
from dataset.transform import train_transform
import pandas as pd
batch_size=4
path_images='C:/Users/lukic/Documents/Диплом/Keypoints/images/'
keypoint_df=pd.read_csv('C:/Users/lukic/Documents/Диплом/Keypoints/mpii_dataset.csv')
mpii_dataset=MPIIDataset(path_images=path_images,
                         keypoints_df=keypoint_df,
                         size=256,
                         transforms=True)
train_dataloder=DataLoader(dataset=mpii_dataset,
                           batch_size=batch_size)
