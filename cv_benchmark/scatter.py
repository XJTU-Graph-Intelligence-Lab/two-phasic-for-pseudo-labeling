import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import cv2
import os
import os.path as osp

from semilearn.datasets import get_cifar
path_root = osp.dirname(osp.abspath(__file__))

## 显示图片的函数
def imshow(img, idx, path):
    npimg = np.array(img[0])
    cv2.imwrite(os.path.join('pics',path,f'{idx}.jpg'), npimg)

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR100(root='/mnt/bn/liyullm2/USB/data/cifar100', train=True,
                                        download=True)

label = torch.tensor(trainset.targets)
training_dynamics = torch.load('/mnt/bn/liyullm2/USB/saved_models/two_phase/pseudolabel/cifar_400/record_epoch4/tp_latest_model.pth')['training_dynamic'].cpu() #  训练动态
pre_labels = training_dynamics[:,:,-1].argmax(1) #  预测label
right_mask = (pre_labels == label).cpu()


##### 散点图 #####
## 计算confidence和variability
confidence = torch.tensor([training_dynamics[i,int(pre_labels[i]),:].mean().item() for i in range(50000)])
variability = torch.tensor([training_dynamics[i,int(pre_labels[i]),:].std().item() for i in range(50000)])

## 排除训练集
train_mask = (training_dynamics[:,:,0].sum(axis=1) == 0).cpu()
right_mask_wo_train = right_mask[~train_mask]
x = confidence[~train_mask].numpy()
y = variability[~train_mask].numpy()

## 存下两类样本
high_confidence = torch.where(torch.logical_and(confidence>0.8, variability<0.3))[0]
low_consistency = torch.where(torch.logical_and(confidence>0.6, confidence<0.8))[0]
# torch.save(high_confidence, 'high_consistency.pt')
# torch.save(low_consistency, 'low_consistency.pt')
high_consistency_acc = right_mask[high_confidence].sum() / len(high_confidence)
low_consistency_acc = right_mask[low_consistency].sum() / len(low_consistency)

# tp = torch.zeros_like(label)
# tp[torch.load(osp.join(path_root, f'cifar10_two_phase_1000_0_15.pt')).tolist()] = 1
# tp = tp.bool().cpu()

plt.clf()
# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

#  散点图
plt.figure(figsize=(16, 12), dpi=900)
ax = plt.subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.scatter(x[right_mask_wo_train], y[right_mask_wo_train], s=1, marker='o', color='#3498DB', alpha=1) #  正确
plt.scatter(x[~right_mask_wo_train], y[~right_mask_wo_train], s=0.5, marker='o', color='#d73027', alpha=1) #  错误
plt.xlabel('Confidence score')
plt.ylabel('Non-stationary')
plt.savefig('scatter.png', dpi=900)

plt.clf()
# 创建图例图
legend_fig, legend_ax = plt.subplots(figsize=(4, 1), dpi=900)
# 创建放大的散点图示例
blue_dot = mlines.Line2D([], [], color='#3498DB', marker='o', linestyle='None', markersize=10, label='Correctly predicted sample')
red_dot = mlines.Line2D([], [], color='#d73027', marker='o', linestyle='None', markersize=10, label='Incorrectly Predicted sample')
# 创建图例
legend_ax.legend([blue_dot, red_dot], ['Correctly predicted sample', 'Incorrectly predicted sample'], loc='center', frameon=False)
# 隐藏坐标轴
legend_ax.axis('off')
# 保存图例图
legend_fig.savefig('legend.png', dpi=900)

## 筛选图片
## 置信度高、预测正确且类别为马
green_right = torch.logical_and(confidence>0.9, right_mask)
green_right_horse = torch.logical_and(green_right, (label==8).cpu())
green_idx = torch.where(green_right_horse)[0]
for i in green_idx[:20]:
    imshow(trainset[i], i, 'green')

## variability较高、预测正确且类别为马
red = torch.logical_and(torch.logical_and(confidence>0.6, confidence<0.8), variability>0.3)
red_right_horse = torch.logical_and(torch.logical_and(red, right_mask), (label==7).cpu())
red_idx = torch.where(red_right_horse)[0]
for i in red_idx[:20]:
    imshow(trainset[i], i, 'red')
