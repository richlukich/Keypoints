from dataset.mpii import MPIIDataset
import torch
from torch.utils.data import DataLoader
from utils.gaussian import generateGaussian
from utils import utils
import pandas as pd
batch_size=8
path_images='/content/images/'
gauss= generateGaussian(utils.output_size,utils.nJoints)
mpii_dataset_train=MPIIDataset('train',
                         gauss)
mpii_dataset_val=MPIIDataset('val',
                         gauss)

train_dataloader=DataLoader(mpii_dataset_train,
                            batch_size=batch_size)

val_dataloader=DataLoader(mpii_dataset_val,
                         batch_size=1)
