import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np
import json
# from tkinter import _flatten

def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)

    x_ts = ts.fit_transform(feat) # 降维

    print(x_ts.shape)  # [num, 2]

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final

# 设置散点形状
# maker = ['o', 's', '^', 'p', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
maker = ['o']
# 设置散点颜色
colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink']
# 图例名称
Label_Com = ['share_A', 'share_T', 'private_A', 'private_T']
# 设置字体格式
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }


def plotlabels(S_lowDWeights, Trure_labels, name):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape)  # [num, 3]

    for index in range(5):  # 假设总共有三个类别，类别的表示为0,1,2
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        if index == 0:
            plt.scatter(X, Y,  label = '$share_A$',cmap='brg', s=100, marker=maker[0], c=colors[index], edgecolors=colors[index], alpha=0.65)
        elif index == 1:
            plt.scatter(X, Y,  label = '$share_T$',cmap='brg', s=100, marker=maker[0], c=colors[index], edgecolors=colors[index], alpha=0.65)
        elif index == 2:
            plt.scatter(X, Y,  label = '$share_V$',cmap='brg', s=100, marker=maker[0], c=colors[index], edgecolors=colors[index], alpha=0.65)
        elif index == 3:
            plt.scatter(X, Y,  label = '$private_A$',cmap='brg', s=100, marker=maker[0], c=colors[index], edgecolors=colors[index], alpha=0.65)
        elif index == 4:
            plt.scatter(X, Y,  label = '$private_T$',cmap='brg', s=100, marker=maker[0], c=colors[index], edgecolors=colors[index], alpha=0.65)
        elif index == 5:
            plt.scatter(X, Y,  label = '$private_V$',cmap='brg', s=100, marker=maker[0], c=colors[index], edgecolors=colors[index], alpha=0.65)
        elif index == 6:
            plt.scatter(X, Y,  label = '$private_V$',cmap='brg', s=100, marker=maker[0], c=colors[index], edgecolors=colors[index], alpha=0.65)
        elif index == 7:
            plt.scatter(X, Y,  label = '$private_V$',cmap='brg', s=100, marker=maker[0], c=colors[index], edgecolors=colors[index], alpha=0.65)

        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值

    plt.title(name, fontsize=32, fontweight='normal', pad=20)

# data = np.load("/home/Qy/autodl/Synthesis/DiffMIC (copy)/dataset/isic2018/isic2018_sne.npy", allow_pickle=True)

# data = np.load("/home/Qy/autodl/Synthesis/DiffMIC (copy)/dataset/aptos/aptos_sne.npy", allow_pickle=True)
data = np.load("/home/Qy/autodl/Synthesis/DiffMIC (copy)/dataset/aptos/aptos_sne_2.0.npy", allow_pickle=True)
datalist = data.tolist()

feat =np.array(datalist['feature'], dtype="float32")
label = np.array(datalist["label"])


fig = plt.figure(figsize=(10, 10))

plotlabels(visual(feat), label, '(a)')

plt.savefig('dataset/aptos/t_sne_2.0.png', dpi=300)
# plt.savefig('dataset/isic2018/t_sne.png', dpi=300)
#plt.show(fig)