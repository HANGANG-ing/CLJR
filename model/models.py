import torch
import torchvision
from torch import nn
from torch import functional as F

from d2l import torch as d2l
import wandb
from torch import nn
from .resnet_cifar import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, DenseNet121, DenseNet161
import torch


class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim
        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        out_dim = in_dim
        hidden_dim = int(out_dim / 4)

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class Linear(nn.Module):
    def __init__(self, nb_classes=10, feat=512):
        super(Linear, self).__init__()
        self.linear = nn.Linear(feat, nb_classes)

    def forward(self, x):
        return self.linear(x)


class BatchNorm1d(nn.Module):
    def __init__(self, dim, affine=True, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine, momentum=momentum)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SimSiam(nn.Module):
    def __init__(self, emam, arch, nb_classes):
        super(SimSiam, self).__init__()
        self.emam = emam

        self.backbone = SimSiam.get_backbone(arch)
        self.backbone_k = SimSiam.get_backbone(arch)
        if (arch == 'densenet121' or arch == 'densenet161'):
            dim_out, dim_in = self.backbone.classifier.weight.shape
            dim_mlp = 2048
            self.backbone.classifier = nn.Identity()
            self.backbone_k.classifier = nn.Identity()
        else:
            dim_out, dim_in = self.backbone.fc.weight.shape
            dim_mlp = 2048
            self.backbone.fc = nn.Identity()
            self.backbone_k.fc = nn.Identity()

        print('dim in', dim_in)
        print('dim out', dim_out)
        self.projector = nn.Sequential(nn.Linear(dim_in, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(),
                                       nn.Linear(dim_mlp, dim_out))
        self.projector_k = nn.Sequential(nn.Linear(dim_in, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(),
                                         nn.Linear(dim_mlp, dim_out))

        # predictor
        self.predictor = nn.Sequential(nn.Linear(dim_out, 64), BatchNorm1d(64), nn.ReLU(), nn.Linear(64, dim_out))
        self.predictor_k = nn.Sequential(nn.Linear(dim_out, 64), BatchNorm1d(64), nn.ReLU(), nn.Linear(64, dim_out))
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        self.encoder_k = nn.Sequential(
            self.backbone_k,
            self.projector_k
        )

        self.linear = Linear(nb_classes=nb_classes, feat=dim_in)
        self.linear_k = Linear(nb_classes=nb_classes, feat=dim_in)
        self.linear2 = Linear(nb_classes=nb_classes, feat=dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.probability = nn.Sequential(
            self.backbone,
            self.linear,
            self.softmax
        )
        self.probability_k = nn.Sequential(
            self.backbone_k,
            self.linear_k,
            self.softmax
        )
        self.classifier_head = nn.Sequential(
            self.encoder_k,
            self.linear2
        )
        for param_q, param_k in zip(self.probability.parameters(), self.probability_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @staticmethod
    def get_backbone(backbone_name):
        return {'resnet18': ResNet18(),
                'resnet34': ResNet34(),
                'resnet50': ResNet50(),
                'resnet101': ResNet101(),
                'resnet152': ResNet152(),
                'densenet121': DenseNet121(),
                'densenet161': DenseNet161(), }[backbone_name]

    def forward(self, im_aug1, im_aug2):  # 输入数据，给出两个预测
        # output = self.probability(img_weak)

        # z1 = self.encoder(im_aug1)
        # p1 = self.predictor(z1)
        # p1 = nn.functional.normalize(p1, dim=1)
        p1 = self.probability(im_aug1)

        with torch.no_grad():  # no gradient to keys
            m = self.emam
            for param_q, param_k in zip(self.probability.parameters(), self.probability_k.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1. - m)

        # z2 = self.encoder_k(im_aug2)
        # z2 = nn.functional.normalize(z2, dim=1)
        p2 = self.probability_k(im_aug2)

        return p1, p2

    def forward_strong(self, im_aug1, im_aug2):
        z1 = self.encoder(im_aug1)
        q1 = self.predictor(z1)
        q1 = nn.functional.normalize(q1, dim=1)
        z1 = nn.functional.normalize(z1, dim=1)
        with torch.no_grad():  # no gradient to keys
            m = self.emam
            for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1. - m)

        z2 = self.encoder(im_aug2)
        q2 = self.predictor(z2)
        q2 = nn.functional.normalize(q2, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        return q1, z2.detach(), q2, z1.detach()

    def forward_test(self, x):
        # return self.probability_k(x)
        return self.probability_k(x)

    def forward_LT(self, x):
        # retrain_classifier for LT

        return self.classifier_head(x)
