from utils.utils import AverageMeter
import torch
def step(model,epoch, dataloader , criterion,optimizer):
    Loss=AverageMeter()
    preds=[]

    nIters=len(dataloader)

    for images,keypoints,targets in dataloader:
        input_var = torch.autograd.Variable(images).float().cuda()
        targets_var=torch.autograd.Variable(targets).float().cuda()

        output=model(input_var)
        loss=criterion(output[0],targets_var)
        Loss.update(loss.item(), input_var.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ('Epoch=',epoch,'Loss=',Loss.avg)

def train(model,epoch, dataloader , criterion,optimizer):
    return step(model,epoch, dataloader , criterion,optimizer)