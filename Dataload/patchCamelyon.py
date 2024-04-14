import h5py
import numpy as np
import torchvision.transforms
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader

from model.utils import noisify


class Camelyonpatch(data.Dataset):
    def __init__(self,noise_type='sym',r=0.2,train=True,transform1=None,transform2=None,random_state=0,nb_classes=2):

        self.train=train
        self.transform1=transform1
        self.transform2=transform2
        self.nb_classes=nb_classes
        # if self.train:
        filex = h5py.File('/home/hangang/datasets/camelyonpath/camelyonpatch_level_2_split_train_x.h5', 'r')
        filey = h5py.File('/home/hangang/datasets/camelyonpath/camelyonpatch_level_2_split_train_y.h5', 'r')
        # else:
        #     filex = h5py.File('/home/hangang/datasets/camelyonpath/camelyonpatch_level_2_split_test_x.h5', 'r')
        #     filey = h5py.File('/home/hangang/datasets/camelyonpath/camelyonpatch_level_2_split_test_y.h5', 'r')

        self.dataset=filex['x']
        self.targets=filey['y']
        #num 131072 262144




        if self.train:
            train_list = np.arange(30000)
            np.random.shuffle(train_list)
            self.dataset_train = [self.dataset[train_list[i]] for i in range(26214)]
            self.targets_train = [self.targets[train_list[i]] for i in range(26214)]
            #print(self.targets_train)
            self.targets_train=np.array(self.targets_train).flatten()
            self.train_labels = np.asarray([[self.targets_train[i]] for i in range(len(self.targets_train))])
            self.train_noisy_labels=self.train_labels
            if r!=0:
                self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset,
                                                                        train_labels=self.train_labels,
                                                                        noise_type=noise_type, noise_rate=r,
                                                                        random_state=random_state,
                                                                        nb_classes=self.nb_classes)

            self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]
            self.train_data=self.dataset_train
            self.train_label=self.targets_train

        else:
            test_list = np.arange(30010, 35000)
            np.random.shuffle(test_list)
            # print(len(self.dataset),'+++++++')

            self.dataset_test = [self.dataset[test_list[i]] for i in range(3276)]
            self.targets_test = [self.targets[test_list[i]] for i in range(3276)]


    def __getitem__(self, index):
        #self.dataset[index]=self.dataset[index].transpose([2, 0,1])
        Images=[]
        if self.train:
            img = Image.fromarray(self.dataset_train[index])
            img1 = self.transform1(img)
            img2=self.transform1(img)
            img3=self.transform2(img)
            img4=self.transform2(img)
            target = self.train_noisy_labels[index].item()
            Images=[img1,img2,img3,img4]

            return Images, target,  index
        else:
            img = Image.fromarray(self.dataset_test[index])
            img = self.transform1(img)
            target = self.targets_test[index].item()
            Images.append(img)
            return Images,target, index

    def __len__(self):
        if self.train:
            return len(self.targets_train)
        else:
            return len(self.targets_test)

# transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# dataset=Camelyonpatch(train=True,transforms=transform)
# dataloader=DataLoader(dataset=dataset,batch_size=128)
# sum=0
# tot=0
# for data in dataloader:
#     x,y,_=data
#     #print(x.shape,y.shape)
#     sum=sum+y.sum()
#     tot=tot+y.size(0)
#     #print(x.size())
# print(sum,tot)