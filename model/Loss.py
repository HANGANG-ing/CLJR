import copy

import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn
from torch.nn.functional import one_hot

from model.Info_NCE import InfoNCE
criterion = nn.CosineSimilarity(dim=1).cuda(0)
import numpy as np

# utils
def contrastive_loss(q, k):
    T=1.0
    # normalize
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)
    # Einstein sum is more intuitive
    logits = torch.einsum('nc,mc->nm', [q, k]) / T
    N = logits.shape[0]  # batch size per GPU
    labels = torch.arange(N, dtype=torch.long).cuda()
    return nn.CrossEntropyLoss()(logits, labels) * (2 * T)


def Loss(epoch, max_epoch, p1, p1_k, Histroy_P1, q1, z2, q2, z1, labels,args):
    a = (1.0 - (float(epoch / max_epoch) ** 4))
    # loss=loss1+a * loss2 + (1 - a) * loss3

    # print(Nlabel)

    # num_classes = args.num_classes
    # pre_label = (a * label + (1 - a) * p1)
    loss_ctr = -(criterion(q1, z2).mean() + criterion(q2, z1).mean()) * 0.5
    # loss_ctr=contrastive_loss(q1,z2)+contrastive_loss(q2,z1)

    if epoch <= args.warm_up:
        loss1 = F.cross_entropy(p1, labels)
        loss = 0.5*loss1+loss_ctr
        # loss = 0.5 * loss1
    else:
        Nlabel = copy.deepcopy(labels)
        bsz = labels.size(0)
        label = one_hot(Nlabel, num_classes=args.num_classes)
        ones = Variable(torch.ones(args.num_classes)).cuda()
        Nlabel = (ones - label) / (args.num_classes - 1)

        JS_div = F.kl_div(p1.log(), p1_k, reduction='none').sum(dim=1)
        # print(KL_div)
        b = (max_epoch-args.warm_up) / (epoch-args.warm_up)
        miu = 0.5*b
        # print(JS_div)
        with torch.no_grad():
            mask1 = Variable(torch.tensor([int(x) for x in JS_div < miu],dtype=torch.float)).cuda()
            mask2 = Variable(torch.tensor([1 - x for x in mask1],dtype=torch.float)).cuda()

            #mask1=Variable(torch.ones(bsz)).cuda()
            #mask2=Variable(torch.zeros(bsz)).cuda()
        # print(mask1, mask2)
        # loss1 = torch.dot(F.cross_entropy(p1, labels, reduction='none') , mask1)/bsz
        # loss11=F.cross_entropy(p1*mask1, labels*mask1)
        mask_p1=p1*mask1.unsqueeze(1)
        mask_label=label*mask1.unsqueeze(1)
        loss1 = F.cross_entropy(mask_p1, mask_label)
        loss11=F.cross_entropy(p1,labels)
        # print(f'loss1{loss1},loss11{loss11}')

        # fu loss
        mask2_p1=(F.softmax(ones-p1,dim=1))*mask2.unsqueeze(1)
        mask2_Nlabel=Nlabel*mask2.unsqueeze(1)
        loss2 = F.cross_entropy(mask2_p1,mask2_Nlabel)

        lossH=F.cross_entropy(p1,Histroy_P1)
        loss3=F.cross_entropy(p1,p1_k)
        lamd=2
        # loss = loss11+loss1+loss2+lossH+loss_ctr+loss3
        loss = loss11+ lamd*(loss1 + loss2 + lossH + loss3+ 0.1*loss_ctr)
        # loss=loss1+lamd*(lossH)
        # print(f'pa:{para_a.requires_grad,para_a},pb:{para_b.requires_grad,para_b}')
        # loss = loss11
    return loss
