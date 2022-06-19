import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import cv2
from utils.gaussian import generate_gaussian
import matplotlib.pyplot as plt
from dataset.transform import train_transform
from h5py import File


class MPIIDataset(Dataset):
    def __init__(self, split, transforms=None, input_size=256, output_size=64, nJoints=16):
        self.split = split
        self.input_size = input_size
        self.output_size = output_size
        self.transforms = transforms
        self.nJoints = nJoints

        self.file = File((f'/content/{split}.h5'))

    def GetImage(self, idx):
        img_name = str(self.file['imgname'][idx])[2:-1]
        image = plt.imread(f'/content/images/{img_name}')
        return image,img_name

    def GetKeypoints(self, idx):
        return self.file['part'][idx]

    def remove_invisible(self, keypoint):
        if keypoint[0] < 0 or keypoint[1] < 0 or keypoint[0] > self.input_size or keypoint[1] > self.input_size:
            return (0, 0)
        else:
            return keypoint

    def __getitem__(self, idx):
        image,img_name = self.GetImage(idx)
        keypoints = self.GetKeypoints(idx)
        if self.transforms:
            image, keypoints = self.transforms(image, keypoints, self.input_size, self.input_size)

            keypoints = list(map(self.remove_invisible, keypoints))
        targets = np.zeros((self.nJoints, self.output_size, self.output_size))
        k = self.output_size / self.input_size
        for i in range(self.nJoints):
            targets[i] = generate_gaussian(targets, keypoints[i][0] * k,
                                               keypoints[i][1] * k)

        meta={'imgname':img_name}
        return image, targets,meta

    def __len__(self):
        return len(self.file['imgname'])