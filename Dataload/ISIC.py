import random

import numpy as np
import torch.utils.data as data
import pandas as pd
import torchvision
from PIL import Image
import os

from torch.utils.data import DataLoader

from model.utils import noisify


# ISIC数据加载
class ISIC(data.Dataset):
    def __init__(self, noise_type='sym', r=0, path_list=None, label_list=None, train=True, transform1=None,transform2=None,transform3=None):
        self.train = train

        self.data_root = '/home/hangang/datasets/ISIC/ISIC_Training_Data'
        self.label_root = '/home/hangang/datasets/ISIC/ISIC_Training_Data.csv'
        self.test_data_root='/home/hangang/datasets/ISIC/ISIC_Test_Data'
        self.test_label_root='/home/hangang/datasets/ISIC/ISIC_Test_Data.csv'
        if self.train:
            # self.noise_targets = corrupted_ISIC(self.ban_targets, r, noise_type)
            self.label_list = pd.read_csv(self.label_root)
            self.data_dir = os.listdir(self.data_root)
            self.data_list = []
            for data in self.data_dir:
                if (data.endswith('.jpg')):
                    self.data_list.append(data)
            self.dataset = 'ISIC'
            self.targets = []
            for index in range(len(self.data_list)):

                self.targets.append(
                    self.label_list.loc[(self.label_list['image_id'] == self.data_list[index][0:12]), 'MEL'].item())


            self.lbph_num = 0
            self.lbph_sum = 0
            self.targets_train = np.array(self.targets).flatten()
            self.train_labels = np.asarray([[int(self.targets_train[i])] for i in range(len(self.targets_train))])
            if r:
                self.noise_targets, self.actual_noise_rate = noisify(dataset=self.dataset,
                                                                 train_labels=self.train_labels,
                                                                 noise_type=noise_type, noise_rate=r,
                                                                 random_state=0,
                                                                 nb_classes=2)
            else:
                self.noise_targets=self.train_labels

            self.train_data = self.data_list
            self.train_noisy_labels = self.noise_targets
            self.train_label = self.targets
            self.noise_or_not = np.transpose(self.noise_targets) == np.transpose(self.targets)
            for data in self.train_label:
                self.lbph_num = self.lbph_num + data
                self.lbph_sum = self.lbph_sum + 1
            print('lbph_sum={},lbph_num={}'.format(self.lbph_sum, self.lbph_num))
        else:
            self.test_data_dir=os.listdir(self.test_data_root)
            self.test_label_list = pd.read_csv(self.test_label_root)
            self.test_data_list=[]
            for data in self.test_data_dir:
                if (data.endswith('.jpg')):
                    self.test_data_list.append(data)
            self.dataset = 'ISIC'
            self.test_targets = []
            for index in range(len(self.test_data_list)):
                #print(self.test_data_list[index][0:-4],"+++",self.test_label_list.loc[(self.test_label_list['image_id'] == self.test_data_list[index][0:-4]), 'melanoma'])
                self.test_targets.append(
                    self.test_label_list.loc[(self.test_label_list['image_id'] == self.test_data_list[index][0:12]), 'melanoma'].item())

        self.transform1 = transform1
        self.transform2=transform2
        self.nb_classes = 2

    def __getitem__(self, index):
        Images=[]
        if self.train:
            img = Image.open(os.path.join(self.data_root, self.data_list[index]))

            img1 = self.transform1(img)
            img2 = self.transform1(img)
            img3 = self.transform2(img)
            img4 = self.transform2(img)
            true_target = int(self.targets[index])
            target = int(self.train_noisy_labels[index])
            Images = [img1, img2, img3, img4]
        else:
            img = Image.open(os.path.join(self.test_data_root, self.test_data_list[index]))

            if self.transform1 is not None:
                img = self.transform1(img)
            target = int(self.test_targets[index])
            Images.append(img)
        return Images, target, index

    def __len__(self):
        if self.train:
            return len(self.data_list)
        else:
            return len(self.test_data_list)

# transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize((224,224))])
# isic=ISIC2017(transform1=transform,train=False)
# isic_loader=DataLoader(isic,batch_size=32)
# print(len(isic_loader))
# for data in isic_loader:
#     img,target,_=data
#     print(img.shape)
#     print(target)
