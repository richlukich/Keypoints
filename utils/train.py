from utils.evualate import AverageMeter
import time
from utils.evualate import Accuracy
import torch
def train(train_loader,model,criterion,optimizer,epoch):

    Loss=AverageMeter()
    Acc=AverageMeter()
    data_time=AverageMeter()
    batch_time=AverageMeter()
    model.train()
    end=time.time()

    for i,(input,targets,meta) in train_loader:
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input).float().cuda()
        target_var = torch.autograd.Variable(targets).float().cuda()

        outputs=model(input_var)

        loss=criterion(outputs,target_var)

        optimizer.zero_grad
        loss.backward()
        optimizer.step()

        Loss.update(loss.item(), input.shape[0])

        accuracy=Accuracy(outputs.detach().cpu().numpy(),
                     target_var.detach().cpu().numpy())

        Acc.update(accuracy)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 300 == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.shape[0] / batch_time.val,
                data_time=data_time, loss=Loss, acc=Acc)