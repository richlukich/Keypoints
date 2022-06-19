from dataset.mpii import MPIIDataset
import torch
from torch.utils.data import DataLoader
from dataset.transform import train_transform,val_transform
import pandas as pd
batch_size=4
path_images='/content/images/'
mpii_dataset_train=MPIIDataset('train',
                         train_transform)
mpii_dataset_val=MPIIDataset('val',
                         val_transform)

train_dataloader=DataLoader(mpii_dataset_train,
                            batch_size=batch_size)

val_dataloder=DataLoader(mpii_dataset_val,
                         batch_size=1)
