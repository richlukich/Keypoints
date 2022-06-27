from torch.utils.data import Dataset
import numpy as np
import torch
from h5py import File
import cv2
import matplotlib.pyplot as plt
from utils import utils
from utils.gaussian import generateGaussian
from dataset import transform
from dataset.transform import crop,kpt_affine
class MPIIDataset(Dataset):
    def __init__(self, split, generateGaussian, input_size=utils.input_size, output_size=utils.output_size):
        print('==> initializing 2D {} data.'.format(split))
        annot = {}
        tags = ['imgname', 'part', 'center', 'scale']
        f = File('{}/mpii/annot/{}.h5'.format(utils.annotDir, split), 'r')
        for tag in tags:
            annot[tag] = np.asarray(f[tag]).copy()
        f.close()

        print('Loaded 2D {} {} samples'.format(split, len(annot['scale'])))

        self.split = split
        self.annot = annot
        self.input_size = input_size
        self.output_size = output_size
        self.generateGaussian = generateGaussian

    def LoadImage(self, index):
        imgname = str(self.annot['imgname'][index])[2:-1]
        path = '{}/{}'.format(utils.img_dir, imgname)
        img = plt.imread(path)
        return img

    def GetPartInfo(self, index):
        pts = self.annot['part'][index].copy()
        c = self.annot['center'][index].copy()
        s = self.annot['scale'][index]
        return pts, c, s

    def __getitem__(self, index):
        img = self.LoadImage(index)
        pts, c, s = self.GetPartInfo(index)

        cropped = crop(img, c, s, (self.input_size, self.input_size)) / 255

        keypoints = np.zeros((np.shape(pts)))
        for i in range(np.shape(pts)[0]):
            if np.sum(pts[i]) > 0:
                keypoints[i] = transform(pts[i], c, s, (self.input_size, self.input_size))

        height, width = cropped.shape[0:2]
        center = np.array((width / 2, height / 2))
        scale = max(height, width) / 200

        aug_rot = 0
        if self.split == 'train:':
            aug_rot = (np.random.random() * 2 - 1) * 30.
            aug_scale = np.random.random() * (1.25 - 0.75) + 0.75
            scale *= aug_scale

        else:
            scale *= 1.25
        mat_mask = get_transform(center, scale, (self.output_size, self.output_size), aug_rot)[:2]

        mat = get_transform(center, scale, (self.input_size, self.input_size), aug_rot)[:2]
        inp = cv2.warpAffine(cropped, mat, (self.input_size, self.input_size)).astype(np.float32)
        keypoints = kpt_affine(keypoints, mat_mask)

        for i in range(np.shape(pts)[0]):
            if pts[i][0] == 0 and pts[i][1] == 0:
                keypoints[i][0] = 0
                keypoints[i][1] = 0

        heatmaps = self.generateGaussian(keypoints)

        if self.split == 'train':
            #  if np.random.random() < 0.5:
            #    inp = Flip(inp)
            #    heatmaps = ShuffleLR(Flip(heatmaps))

            inp[0] = np.clip(inp[0] * (np.random.random() * (0.4) + 0.6), 0, 1)
            inp[1] = np.clip(inp[1] * (np.random.random() * (0.4) + 0.6), 0, 1)
            inp[2] = np.clip(inp[2] * (np.random.random() * (0.4) + 0.6), 0, 1)
            ## generate heatmaps on outres
            meta = np.zeros(1)
        else:
            meta = {'index': index, 'center': c, 'scale': s, 'rotate': aug_rot}
        return np.rollaxis(inp.astype(np.float32), 2, 0), heatmaps.astype(np.float32), meta

    def __len__(self):
        return len(self.annot['scale'])
