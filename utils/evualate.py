import numpy as np
def getPreds(heatmaps):

    assert isinstance(heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size=heatmaps.shape[0]
    nJoints=heatmaps.shape[1]
    width=height=heatmaps.shape[3]

    heatmaps_vectors=heatmaps.reshape(batch_size,nJoints,width*height)
    maxidx=np.argmax(heatmaps_vectors,axis=2)
    maxvalue=np.amax(heatmaps_vectors,axis=2)

    preds=np.tile(maxidx, (1, 1, 2)).astype(np.float32)

    preds[:,:,0]=preds[:,:,0] % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvalue, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds*=pred_mask
    return preds

def CalcDist(output,target,normalize=64/10):

    dists=np.zeros((output.shape[0],output.shape[1]))

    for i in range (dists.shape[0]):
        for j in range (dists.shape[1]):

            if target[i,j,0]>0 and target[i,j,1]>0:
                dists[j][i]=( ((output[i,j,0]-target[i,j,0])**2 +  (output[i,j,1]-target[i,j,1])**2)**0.5 ) /normalize
            else:
                dists[j][i]=-1
    return dists

def dists_acc(dists,threshold=0.5):
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal>0:
        return np.less(dists[dist_cal], threshold).sum() * 1.0 / num_dist_cal
    else:
        return -1

def Accuracy(output,targets):
    preds=getPreds(output)
    targets=getPreds(targets)
    avg_sum=0
    bad_idx = 0

    dists=CalcDist(preds,targets)
    acc_idx=list(np.arange(16))
    accuracy=np.zeros(len(acc_idx))

    for i in acc_idx:
        accuracy[i]=dists_acc(dists[i])
        if accuracy[i] >= 0:
            avgAcc = avg_sum + accuracy[i]
        else:
            bad_idx = bad_idx + 1
    if bad_idx==len (acc_idx):
        return 0
    else:
        return avg_sum/(len(acc_idx) - bad_idx)


class AverageMeter(object):
    """Computes and stores the average and current value"""
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