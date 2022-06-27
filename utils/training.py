# -*- coding:utf-8 -*-
import torch
import numpy as np
from utils.evualate import AverageMeter
from utils.evualate import Accuracy,Accuracy_dark,Accuracy_centroid,Accuracy_merge
import cv2
from utils import utils
from progress.bar import Bar


def step(split, epoch, dataLoader, model, criterion, mode,optimizer = None):
    if split == 'train':
        model.train()
    else:
        model.eval()
    Loss, Acc = AverageMeter(), AverageMeter()
    preds = []

    nIters = len(dataLoader)
    bar = Bar('{}'.format('default'), max=nIters)
    for i, (input, target, meta) in enumerate(dataLoader):
        input_var = torch.autograd.Variable(input).float().cuda()
        target_var = torch.autograd.Variable(target).float().cuda()
        output = model(input_var)



        loss = criterion(output, target_var)
        Loss.update(loss.item(), input.shape[0])
        if split=='val':
          if mode=="argmax":
            Acc.update(Accuracy((output.data).cpu().numpy(), (target_var.data).cpu().numpy()))
          if mode=="dark":
            Acc.update(Accuracy_dark((output.data).cpu().numpy(), (target_var.data).cpu().numpy()))
          if mode=="centroid": 
            Acc.update(Accuracy_centroid((output.data).cpu().numpy(), (target_var.data).cpu().numpy()))
          if mode=="merge": 
            Acc.update(Accuracy_merge((output.data).cpu().numpy(), (target_var.data).cpu().numpy()))
        else:
          Acc.update(Accuracy((output.data).cpu().numpy(), (target_var.data).cpu().numpy()))
        if split == 'train':
            # train
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i%300==0:
            print ('{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Acc {Acc.avg:.6f} ({Acc.val:.6f})'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split))
        Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Acc {Acc.avg:.6f} ({Acc.val:.6f})'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split)
        bar.next()

    bar.finish()
    return {'Loss': Loss.avg, 'Acc': Acc.avg}, preds


def train(epoch, train_loader, model, criterion, mode,optimizer):
    return step('train', epoch, train_loader, model, criterion,mode, optimizer)

def val(epoch, val_loader, model, criterion,mode):
    return step('val', epoch,  val_loader, model, criterion,mode)
