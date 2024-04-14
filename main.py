# -*- coding:utf-8 -*-
import os
from typing import List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from Dataload.ISIC import ISIC
from Dataload.patchCamelyon import Camelyonpatch
# from Dataload.ISIC import ISIC
from Dataload.Chaoyang import CHAOYANG
import argparse, sys
import numpy as np
import torchvision.models as models
import pickle

from sklearn.metrics import roc_auc_score

from model import utils
from model.Loss import Loss
from model.models import SimSiam

import wandb
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=0.01)
parser.add_argument('--dataset', type=str, default='patchcamelyon')
parser.add_argument('--n_epoch', type=int, default=150)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=16, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=18)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--warm_up', type=int, default=10)
parser.add_argument('--r', type=float, default=0.0)
parser.add_argument('--noise_type', type=str, default='sym')
parser.add_argument('--backbone', type=str, default='densenet121')
parser.add_argument('--s_r', type=float, default='0.2')
parser.add_argument('--m', type=float, default='0.9')
parser.add_argument('--num_classes', type=int, default=2)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters

learning_rate = args.lr

if args.dataset == 'chaoyang':  # 85.97%
    input_channel = 3
    args.num_classes = 4
    args.epoch_decay_start = 30
    args.warm_up = 100
    args.n_epoch = 270
    batch_size = 32
    # train_dataset = pickle.load(open(args.pickle_path,"rb"))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        # transforms.RandomResizedCrop(128, scale=(0.4, 1.)),
        transforms.Resize((256,256)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    train_transforms1 = transforms.Compose(augmentation)
    train_transforms2 = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize((256,256)),
         transforms.ToTensor()])

    train_dataset = CHAOYANG(root="/home/hangang/datasets/chaoyang-data",
                             json_name="train.json",
                             train=True,
                             transform1=train_transforms1,
                             transform2=train_transforms2
                             )
    test_dataset = CHAOYANG(root="/home/hangang/datasets/chaoyang-data",
                            json_name="test.json",
                            train=False,
                            transform1=transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()]),
                            )
elif args.dataset == 'patchcamelyon':  # r=0:93.83% r=0.05=92.09，r=0.1=93.13,r=0.2=87.33,r=0.3=85.99,r=0.4=84.95
    input_channel = 3
    args.num_classes = 2
    args.epoch_decay_start = 50
    args.warm_up = 15
    args.n_epoch = 150
    batch_size = 128

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        # transforms.RandomResizedCrop(96, scale=(0.4, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        # transforms.Resize((96,96)),
        transforms.ToTensor(),
    ]
    train_transforms1 = transforms.Compose(augmentation)

    train_transforms2 = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])

    train_dataset = Camelyonpatch(noise_type=args.noise_type, r=args.r, train=True, transform1=train_transforms1,
                                  transform2=train_transforms2)
    test_dataset = Camelyonpatch(train=False,
                                 transform1=transforms.Compose([transforms.ToTensor()]))

elif args.dataset == 'ISIC':
    input_channel = 3
    args.num_classes = 2
    args.epoch_decay_start = 30
    args.warm_up = 100
    args.n_epoch = 270
    batch_size = 32
    # train_dataset = pickle.load(open(args.pickle_path,"rb"))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        # transforms.RandomResizedCrop(128, scale=(0.4, 1.)),
        transforms.Resize((224,224)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    train_transforms1 = transforms.Compose(augmentation)
    train_transforms2 = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize((224,224)),
         transforms.ToTensor()])

    train_dataset = ISIC(noise_type=args.noise_type, r=args.r, train=True, transform1=train_transforms1,
                                  transform2=train_transforms2)
    test_dataset = ISIC(train=False,
                                 transform1=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]))



# Histroy_P1: list[Any]=[]

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)  # Only change beta1

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# python sklearn包计算auc
def get_auc(y_labels, y_scores):
    auc = roc_auc_score(y_labels, y_scores)
    # print('AUC calculated by sklearn tool is {}'.format(auc))
    return auc

def train(epoch, max_epoch, model_net1, train_load, optim1):
    train_total = 0
    train_correct = 0
    train_total2 = 0
    train_correct2 = 0
    list_p1 = None
    list_p2 = None
    max_acc = 0
    sum_loss = 0
    lenx = 0
    global Histroy_P1
    # print(Histroy_P1)

    for Imag_list, labels, index in tqdm(train_load):
        x_strong1, x_strong2, x_weak1, x_weak2 = Imag_list[0], Imag_list[1], Imag_list[2], Imag_list[3]

        x_strong1, x_weak1, labels = Variable(x_strong1).cuda(), Variable(x_weak1).cuda(), Variable(labels).cuda()
        x_strong2, x_weak2 = Variable(x_strong2).cuda(), Variable(x_weak2).cuda()
        p1, p1_k = model_net1(x_weak1, x_weak2)
        q1, z2, q2, z1 = model_net1.forward_strong(x_strong1, x_strong2)
        if epoch > args.warm_up:
            with torch.no_grad():
                if list_p1 is not None:
                    list_p1 = torch.cat([list_p1, p1], dim=0)
                else:
                    list_p1 = p1
                len1 = len(p1)

            if epoch == args.warm_up + 1:
                loss = Loss(epoch, max_epoch, p1, p1_k, p1, q1, z2, q2, z2, labels, args)
            else:
                with torch.no_grad():
                    # PP = Histroy_P1[lenx:lenx + len1]
                    PP1 = args.m * Histroy_P1[lenx:lenx + len1] + (1 - args.m) * p1  ## shape(32,4)
                # print(lenx, lenx + len1, PP1.shape, p1.shape)
                loss = Loss(epoch, max_epoch, p1, p1_k, PP1, q1, z2, q2, z2, labels, args)
        else:
            loss=Loss(epoch, max_epoch, p1, p1_k, p1, q1, z2, q2, z2, labels, args)
        prec1, _ = accuracy(p1, labels, topk=(1, 1))
        train_total += 1
        train_correct += prec1
        train_total2 += 1
        # q1,q2,z1,z2=0,0,0,0
        # loss = Loss(epoch, max_epoch, p1, p1_k, Histroy_P1, q1, z2, q2, z2, labels, args)
        optim1.zero_grad()
        # optim2.zero_grad()
        loss.backward()
        # loss2.backward()
        optim1.step()
        # optim2.step()
        sum_loss += loss.item()
        # lenx += len1

    if epoch == args.warm_up + 1:
        with torch.no_grad():
            Histroy_P1 = list_p1

    elif epoch > args.warm_up + 1:
        with torch.no_grad():
            Histroy_P1 = args.m * Histroy_P1 + (1 - args.m) * list_p1
        print('H_P_size:', Histroy_P1.shape,Histroy_P1.requires_grad)
        print('ListP1_size:', list_p1.shape,list_p1.requires_grad)
    train_acc1 = float(train_correct) / float(train_total)
    print(f'epoch:{epoch}\tSum_loss:{sum_loss}\tTrain_acc1:{train_acc1}')
    return train_acc1, sum_loss


def evaluate(val_load, model_net1_kd):
    print("Evaluate......")
    model_net1_kd.eval()
    # model_net2_kd.eval()
    sum = 0
    num = 0
    for x, y, index in tqdm(val_load):
        with torch.no_grad():
            x = Variable(x[0]).cuda()
            y = Variable(y).cuda()
            y_hat1 = model_net1_kd.forward_test(x)
            # y_hat2 = model_net2_kd.forward_test(x)
            # y_hat = (y_hat2 + y_hat1) / 2.0
            y_hat = y_hat1
            sum += y_hat.shape[0]
            num += torch.sum(y_hat.argmax(dim=1).type(y.dtype) == y)

    test_acc = float(num / sum)

    return test_acc


def main():
    wandb.login(key='746701c2a8e76fe7f8fce15a867bc973bf30df8e')
    config = dict(
        momentum=0.9,
        architecture=args.backbone,
        dataset_id=args.dataset,
        noise_radio=args.r,
    )
    wandb.init(
        project='Work-2',
        config=config,
        name=f'SSF_{args.dataset}_{args.r}_{args.backbone}_Beg'
    )
    torch.cuda.set_device(0)
    cudnn.benchmark = True

    train_load1 = DataLoader(train_dataset, batch_size, shuffle=True,num_workers=8,pin_memory=True)
    test_load = DataLoader(test_dataset, batch_size, shuffle=False,num_workers=8,pin_memory=True)

    model_net1 = SimSiam(0.99, args.backbone, args.num_classes)
    #model_net1.load_state_dict(torch.load(f'model/log/warm_up{args.dataset}_{args.r}_sd.pth'))
    # model_net2 = model_net1
    model_net1.cuda()
    # model_net2.cuda()

    # for param_q, param_k in zip(model_net1.probability.parameters(), model_net2.probability.parameters()):
    #     print(f'param_q.requires_grad{param_q.requires_grad}')
    #     print(f'param_k.requires_grad{param_k.requires_grad}')
    # para_a = Variable(torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)).cuda()
    # para_b = Variable(torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)).cuda()
    #
    # params = [
    #     {
    #         'params': model_net1.parameters()
    #     },
    #     {
    #         'params': [para_a, para_b],
    #         'lr': 2e-5
    #     }
    # ]
    optim1 = torch.optim.SGD(model_net1.parameters(), lr=learning_rate, momentum=0.9)
    # optim2 = torch.optim.SGD(model_net1.parameters(), lr=learning_rate, momentum=0.9)
    Histroy_P2 = None
    max_acc = 0

    #for epoch in range(args.warm_up+1,args.n_epoch):
    for epoch in range(args.n_epoch):
        # print(epoch,args.warm_up,"xxxxx")
        model_net1.train()
        # model_net2.train()
        adjust_learning_rate(optim1, epoch)
        train_acc1, train_loss = train(epoch + 1, args.n_epoch, model_net1,
                                                   # model_net2,
                                                   train_load1, optim1,
                                                   # optim2,
                                                   # Histroy_P1,
                                                   )
        test_acc = evaluate(test_load, model_net1,
                            # model_net2,
                            )

        if test_acc > max_acc:
            max_acc = test_acc
            torch.save(model_net1.state_dict(), f'model/log/best{args.dataset}_{args.r}_sd.pth')
        if epoch==args.warm_up:
            torch.save(model_net1.state_dict(),f'model/log/warm_up{args.dataset}_{args.r}_sd.pth')
        print(f'Test_ACC:{test_acc}\tMAX_ACC:{max_acc}')
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_acc1': train_acc1,
                   'test_acc': test_acc, 'max_acc': max_acc})


    with open('model/log/log.txt', 'a') as f:
        f.write(
            'dataset:{}\tnoise_type:{}\tnoise_radio:{}\tbest_acc:{}\t\n'.format(args.dataset, args.noise_type, args.r,
                                                                                max_acc)
        )
    wandb.finish()


if __name__ == '__main__':
    main()
