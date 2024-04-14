import torch.utils.data as data
from PIL import Image
import os
import json
import pickle
import numpy as np
import torch


# from .utils import noisify


class CHAOYANG(data.Dataset):
    def __init__(self, root, json_name=None, path_list=None, label_list=None, train=True, transform1=None,
                 transform2=None):
        imgs = []
        labels = []
        if json_name:
            json_path = os.path.join(root, json_name)
            with open(json_path, 'r') as f:
                load_list = json.load(f)
                for i in range(len(load_list)):
                    img_path = os.path.join(root, load_list[i]["name"])
                    imgs.append(img_path)
                    labels.append(load_list[i]["label"])
        if (path_list and label_list):
            imgs = path_list
            labels = label_list
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train  # training set or test set
        self.dataset = 'chaoyang'

        self.nb_classes = 4
        if self.train:
            # class_num = []
            #
            # for i in range(self.nb_classes):
            #     class_ = []
            #     class_num.append(class_)
            # for i in range(len(labels)):
            #     class_num[labels[i]].append(i)
            # minn = 10000
            # for i in range(self.nb_classes):
            #     minn = min(minn, len(class_num[i]))
            # ban_labels = []
            # ban_imgs = []
            # label_num = []
            # for i in range(self.nb_classes):
            #     label_num = label_num + class_num[i][0:minn - 1]
            #
            # for i in range(len(label_num)):
            #     ban_imgs.append(imgs[i])
            #     ban_labels.append(labels[i])
            # print('minn={},len_ban_img={}'.format(minn, len(ban_imgs)))
            # self.train_data, self.train_labels = ban_imgs, ban_labels
            self.train_data, self.train_labels = imgs, labels
            self.train_noisy_labels = [i for i in self.train_labels]
            self.noise_or_not = [True for i in range(self.__len__())]
        else:
            self.test_data, self.test_labels = imgs, labels

    def __getitem__(self, index):
        Img_list=[]
        if self.train:
            img, target = self.train_data[index], self.train_noisy_labels[index]
            img = Image.open(img)
            if self.transform1 is not None:
                img1 = self.transform1(img)
                img2=self.transform1(img)
                Img_list.append(img1)
                Img_list.append(img2)

            if self.transform2 is not None:
                img3 = self.transform2(img)
                img4 = self.transform2(img)
                Img_list.append(img3)
                Img_list.append(img4)
        else:
            img, target = self.test_data[index], self.test_labels[index]
            img = Image.open(img)
            if self.transform1 is not None:
                img1 = self.transform1(img)
                Img_list.append(img1)

        return Img_list, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
