import numpy as np
from utils import utils
import cv2
def calcDists(preds, gt, normalize):
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if gt[i, j, 0] > 0 and gt[i, j, 1] > 0:
                dists[j][i] = ((gt[i][j] - preds[i][j]) ** 2).sum() ** 0.5 / normalize[i]
            else:
                dists[j][i] = -1
    return dists

def distAccuracy(dist, thr = 0.5):
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum() / len(dist)
    else:
        return -1

def getPreds(heatmap):
    assert len(heatmap.shape) == 4, 'Input must be a 4-D tensor'
    N=heatmap.shape[0]
    C=heatmap.shape[1]
    H=W=heatmap.shape[2]
    hm = heatmap.reshape(N, C, W * H)
    idx = np.argmax(hm, axis=2)
    preds = np.zeros((N, C, 2))
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            preds[i, j, 0], preds[i, j, 1] = idx[i, j] % W, idx[i, j] / H

    return preds


def Accuracy(output, target):
    preds = getPreds(output)
    gt = getPreds(target)
    dists = calcDists(preds, gt, np.ones(preds.shape[0]) * utils.output_size / 10)
    acc = np.zeros(len(utils.accIdxs))
    avgAcc = 0
    badIdxCount = 0

    for i in range(len(utils.accIdxs)):
        acc[i] = distAccuracy(dists[utils.accIdxs[i]])
        if acc[i] >= 0:
            avgAcc = avgAcc + acc[i]
        else:
            badIdxCount = badIdxCount + 1

    if badIdxCount == len(utils.accIdxs):
        return 0
    else:
        return avgAcc / (len(utils.accIdxs) - badIdxCount)


def gaussian_blur(hm, kernel):
    border = (kernel - 1) // 2
    height = hm.shape[0]
    width = hm.shape[1]
    origin_max = np.max(hm)
    dr = np.zeros((height + 2 * border, width + 2 * border))
    dr[border: -border, border: -border] = hm.copy()
    dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
    hm= dr[border: -border, border: -border].copy()
    hm *= origin_max / np.max(hm)
    return hm


def getPreds_dark(target):
    assert len(target.shape) == 4, 'Input must be a 4-D tensor'
    res = target.shape[2]
    N= target.shape[0]
    C=target.shape[1]
    preds=np.zeros((N,C,2))
    for i in range (N):
      for j in range (C):
        hm=target[i,j]
        mask=np.tile(np.greater(hm, 0),(1,1)).astype(np.float32)
        hm=mask*hm
        hm = gaussian_blur(hm, 11)
        hm = np.maximum(hm, 1e-10)
        hm=np.log(hm)
        idx_max=np.argmax(hm)
        coord_x= int(idx_max % 64 )
        coord_y=int (idx_max / 64 )
        #print (coord_x,coord_y)
        if 1 < coord_x < hm.shape[0]-2 and 1 < coord_y < hm.shape[1]-2:
          dx=(hm[coord_y][coord_x+1]-hm[coord_y][coord_x-1]) /2
          dy=(hm[coord_y+1][coord_x]-hm[coord_y-1][coord_x]) /2
          dxx=(hm[coord_y][coord_x-2]-2*hm[coord_y][coord_x]+hm[coord_y][coord_x+2]) / 4
          dyy=(hm[coord_y-2][coord_x]-2*hm[coord_y][coord_x]+hm[coord_y+2][coord_x]) / 4
          dxy=(hm[coord_y+1][coord_x+1]-hm[coord_y-1][coord_x+1]-hm[coord_y+1][coord_x-1]+hm[coord_y-1][coord_x-1]) / 4
          hessian=np.matrix([[dxx,dxy],[dxy,dyy]])
          derivative=np.matrix([[dx,dy]])
          #print (np.linalg.inv(hessian),derivative)
          if dxx * dyy - dxy ** 2 != 0:
            offset=np.linalg.inv(hessian) @ derivative.T
            #print (offset)
            preds[i,j,0],preds[i,j,1]=coord_x - offset[0] , coord_y - offset[1]
          else:
            preds[i,j,0],preds[i,j,1]=coord_x, coord_y
        else:
            preds[i,j,0],preds[i,j,1]=coord_x, coord_y
    return preds


def Accuracy_dark(output, target):
    preds = getPreds_dark(output)
    gt = getPreds_dark(target)
    dists = calcDists(preds, gt, np.ones(preds.shape[0]) * utils.outputRes / 10)
    acc = np.zeros(len(utils.accIdxs))
    avgAcc = 0
    badIdxCount = 0

    for i in range(len(utils.accIdxs)):
        acc[i] = distAccuracy(dists[utils.accIdxs[i]])
        if acc[i] >= 0:
            avgAcc = avgAcc + acc[i]
        else:
            badIdxCount = badIdxCount + 1

    if badIdxCount == len(utils.accIdxs):
        return 0
    else:
        return avgAcc / (len(utils.accIdxs) - badIdxCount)


def getPreds_centroid(target):
    assert len(target.shape) == 4, 'Input must be a 4-D tensor'
    res = target.shape[2]
    N = target.shape[0]
    C = target.shape[1]
    preds = np.zeros((N, C, 2))
    for i in range(N):
        for j in range(C):
            hm = target[i, j]
            mask = np.tile(np.greater(hm, 0), (1, 1)).astype(np.float32)
            hm = mask * hm
            znam = np.sum(hm)
            y_i = x_i = np.arange(64)
            sum_x = sum_y = 0
            for k in range(64):
                I_x = hm[k, :]
                I_y = hm[:, k]
                sum_x += np.sum(I_x * x_i)
                sum_y += np.sum(I_y * y_i)

            preds[i, j, 0], preds[i, j, 1] = sum_x / znam, sum_y / znam
    return preds


def Accuracy_centroid(output, target):
    preds = getPreds_centroid(output)
    gt = getPreds_centroid(target)
    dists = calcDists(preds, gt, np.ones(preds.shape[0]) * utils.outputRes / 10)
    acc = np.zeros(len(utils.accIdxs))
    avgAcc = 0
    badIdxCount = 0

    for i in range(len(utils.accIdxs)):
        acc[i] = distAccuracy(dists[utils.accIdxs[i]])
        if acc[i] >= 0:
            avgAcc = avgAcc + acc[i]
        else:
            badIdxCount = badIdxCount + 1

    if badIdxCount == len(utils.accIdxs):
        return 0
    else:
        return avgAcc / (len(utils.accIdxs) - badIdxCount)


def getPreds_merge(target):
    assert len(target.shape) == 4, 'Input must be a 4-D tensor'
    res = target.shape[2]
    N = target.shape[0]
    C = target.shape[1]
    preds = np.zeros((N, C, 2))
    for i in range(N):
        for j in range(C):
            hm = target[i, j]
            mask = np.tile(np.greater(hm, 0), (1, 1)).astype(np.float32)
            hm = mask * hm
            idx_max = np.argmax(hm)
            coord_x = int(idx_max % 64)
            coord_y = int(idx_max / 64)
            if coord_x > 0 and coord_x < 63 and coord_y > 0 and coord_y < 63:
                dx = (hm[coord_y][coord_x + 1] - hm[coord_y][coord_x - 1])
                dy = (hm[coord_y + 1][coord_x] - hm[coord_y - 1][coord_x])
                preds[i, j, 0] = coord_x + 0.25 * (dx if dx >= 0 else -dx) + 0.5
                preds[i, j, 1] = coord_y + 0.25 * (dy if dy >= 0 else -dy) + 0.5
    return preds


def Accuracy_merge(output, target):
    preds = getPreds_merge(output)
    gt = getPreds_merge(target)
    dists = calcDists(preds, gt, np.ones(preds.shape[0]) * utils.outputRes / 10)
    acc = np.zeros(len(utils.accIdxs))
    avgAcc = 0
    badIdxCount = 0

    for i in range(len(utils.accIdxs)):
        acc[i] = distAccuracy(dists[utils.accIdxs[i]])
        if acc[i] >= 0:
            avgAcc = avgAcc + acc[i]
        else:
            badIdxCount = badIdxCount + 1

    if badIdxCount == len(utils.accIdxs):
        return 0
    else:
        return avgAcc / (len(utils.accIdxs) - badIdxCount)

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count