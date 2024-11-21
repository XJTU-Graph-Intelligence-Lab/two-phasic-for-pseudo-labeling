import os
import torch
import json
import argparse

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader

from semilearn.core.utils import get_net_builder, get_dataset


def entropy(p):
    log = torch.log(p)
    log = torch.where(torch.isinf(log), torch.full_like(log, 0), log)
    entropy = torch.sum(torch.mul(-p, log), dim=1)
    return entropy


def caculate_tp(args, training_dynamic, delta_softmax, pre_label):
    non_zero_mask = (training_dynamic.sum(0).sum(0)!=0)
    training_dynamic = training_dynamic[:,:,non_zero_mask]

    count = training_dynamic.shape[-1]
    sum_softmax = (training_dynamic.sum(-1) / count)

    sum_entropy = entropy(sum_softmax)
    delta_tp, top2_idx = torch.topk(sum_softmax, 2, dim=-1)

    twin_peak_index = torch.zeros_like(top2_idx)
    twin_peak_index[:, 0] = pre_label
    # 创建布尔掩码，检查 pre_labels 是否在 top2_idx 中
    is_first = (pre_label.unsqueeze(1).tile(2) == top2_idx)
    # 选择第二列的值, is_first[:, 0] == True代表pre和top2_idx第一列一致，选第二列，反之选第一列
    twin_peak_index[:, 1] = torch.where(is_first[:, 0], 
                                    top2_idx[:, 1], 
                                    top2_idx[:, 0]) 
    # 处理未匹配的情况, 选top1
    twin_peak_index[~is_first.any(dim=1), 1] = top2_idx[~is_first.any(dim=1), 0]

    delta_not_tp = delta_softmax.sum(dim=1) - torch.gather(delta_softmax, 1, top2_idx).sum(1)
    time_twin_peak = delta_tp.sum(1) / 2
    time_not_twin_peak = delta_not_tp / (args.num_classes-2)
    k_twin_peak = 1 / (1 + torch.exp(-(time_twin_peak - args.b1)))
    k_not_twin_peak = 1 / (1 + torch.exp(-(time_not_twin_peak - args.b2)))
    change_stable = (k_twin_peak)**args.alpha * (k_not_twin_peak)**args.beta

    # 变化存在
    g_final_tmp = torch.gather(training_dynamic[:,:,-1], 1, twin_peak_index.cpu())
    g_final = g_final_tmp[:, 0] - g_final_tmp[:, 1]
    g_min_tmp = torch.gather(training_dynamic, 1, twin_peak_index.unsqueeze(-1).repeat(1, 1, training_dynamic.shape[-1]).cpu())
    g_min = (g_min_tmp[:, 0] - g_min_tmp[:, 1]).min(dim=-1)[0]
    change_exist = (g_final - g_min)+1e-3
    ## 形成指标
    pre_entropy = sum_entropy**args.eta1 * change_stable**args.eta2 * (1/change_exist)**args.eta3
    return pre_entropy.cpu()


class InfoGainCaculater:
    def __init__(self, args, data_type='train_lb'):
        self.net = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes)
        dataset_dict = get_dataset(args, 'pseudolabel', args.dataset, args.num_labels, args.num_classes, args.data_dir, True)
        dataset = dataset_dict[data_type]

        self.data_loader = DataLoader(dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=4)
        self.load_model(args.load_path)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        load_model = checkpoint['ema_model']
        load_state_dict = {}
        for key, item in load_model.items():
            if key.startswith('module'):
                new_key = '.'.join(key.split('.')[1:])
                load_state_dict[new_key] = item
            else:
                load_state_dict[key] = item
        self.net.load_state_dict(load_state_dict)
        self.net.cuda()
    
    def get_all_gain(self):
        all_gain = []
        for data in tqdm(self.data_loader):
            gain = 0
            x = data['x_ulb_w'].cuda()
            y = data['y_ulb'].cuda()

            x.requires_grad_(True)
            output = self.net(x)['logits']  # 假设你的模型是 self.model
            loss = F.cross_entropy(output, y)

            self.net.zero_grad()
            loss.backward()
            
            for _, param in self.net.named_parameters():
                gain += torch.norm(param.grad, p='fro')

            all_gain.append(gain.item())
        return torch.tensor(all_gain)
    
    def get_subset_gain(self, input_x, input_y):
        subset_gain = []
        for idx in tqdm(range(input_x.shape[0])):
            gain = 0
            x = input_x[idx].unsqueeze(0)
            y = input_y[idx].unsqueeze(0)

            x.requires_grad_(True)
            output = self.net(x)['logits']  # 假设你的模型是 self.model
            loss = F.cross_entropy(output, y)

            self.net.zero_grad()
            loss.backward()
            
            for _, param in self.net.named_parameters():
                gain += torch.norm(param.grad, p='fro')

            subset_gain.append(gain.item())
        return torch.tensor(subset_gain)


class CaseStudy3(InfoGainCaculater):
    def __init__(self, args, data_type='train_lb'):
        super().__init__(args, data_type)
        self.y = torch.from_numpy(self.data_loader.dataset.targets).long()
        self.record_td = torch.load(args.tp_load_path)
        self.all_info_gain = torch.load('/mnt/bn/liyullm2/USB/saved_models/two_phase/visual/cifar_400/batch8epoch5/model_24979_21_info_gain.pth')
        self.tp = caculate_tp(args, self.record_td['training_dynamic'], 
                    self.record_td['delta_softmax'].cpu(), self.record_td['pre_label'].cpu())

    @ staticmethod
    def norm_metric(tensor):
        # 计算最小值和最大值
        tensor_min = tensor.min()
        tensor_max = tensor.max()

        # 防止除以零的情况
        if tensor_max - tensor_min == 0:
            return torch.zeros_like(tensor)
        
        # 归一化到 [0, 1]
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        return normalized_tensor
    
    def get_conf_range_info(self):
        # 计算置信度从0-1之间，gap为0.05
        non_zero_mask = (self.record_td['training_dynamic'].sum(0).sum(0)!=0)
        training_dynamic = self.record_td['training_dynamic'][:,:,non_zero_mask]
        confidence, y_hat_c = training_dynamic[:,:,-1].max(-1)
        norm_conf = self.norm_metric(confidence)
        self.show_dist(norm_conf, 'conf_dist.png')

        conf_res = []
        for conf_th in tqdm(np.arange(0, 1, 0.05)):
            pseudo_mask = norm_conf>conf_th
            true_mask = y_hat_c == self.y
            info_gain_all = self.all_info_gain[true_mask].sum() - self.all_info_gain[~true_mask].sum()

            true_num = (y_hat_c[pseudo_mask] == self.y[pseudo_mask]).sum()
            true_prob = true_num/pseudo_mask.sum()
            true_gain = self.all_info_gain[torch.logical_and(true_mask, pseudo_mask)].sum()
            flase_gain = self.all_info_gain[torch.logical_and(~true_mask, pseudo_mask)].sum()
            info_gain = true_gain - flase_gain
            conf_res.append([
                {
                   'th':conf_th, 'true_num': true_num.item(), 'true_prob': true_prob.item(), 'info_gain': info_gain.item()
                }
            ])
        print(info_gain_all)
        return conf_res
    
    def get_tp_range_info(self):
        y_hat = self.record_td['pre_label'].cpu()
        norm_tp = self.norm_metric(-F.sigmoid(self.tp)) # 改变方向，和confidence一样越大越好
        self.show_dist(norm_tp, 'tp_dist.png')
        
        tp_res = []
        for tp_th in tqdm(np.arange(0, 1., 0.05)):
            pseudo_mask = norm_tp>tp_th
            true_mask = y_hat == self.y
            info_gain_all = self.all_info_gain[true_mask].sum() - self.all_info_gain[~true_mask].sum()

            true_num = (y_hat[pseudo_mask] == self.y[pseudo_mask]).sum()
            true_prob = true_num/pseudo_mask.sum()
            true_gain = self.all_info_gain[torch.logical_and(true_mask, pseudo_mask)].sum()
            flase_gain = self.all_info_gain[torch.logical_and(~true_mask, pseudo_mask)].sum()
            info_gain = true_gain - flase_gain
            tp_res.append([
                {
                    'th':tp_th, 'true_num': true_num.item(), 'true_prob': true_prob.item(), 'info_gain': info_gain.item()
                }
            ])
        print(tp_res[-1])
        return tp_res
    
    def show_dist(self, norm_tensor, save_name):
        plt.clf()
        norm_array = norm_tensor.numpy()
        # 设置分箱的边界，0 到 1，间隔为 0.1
        bins = np.arange(0, 1.01, 0.01)  # 生成 0, 0.1, 0.2, ..., 1 的边界

        # 计算每个区间的频数
        counts, bin_edges = np.histogram(norm_array, bins=bins)

        # 绘制柱状图
        plt.bar(bins[:-1], counts, width=0.01, align='edge', edgecolor='black')

        # 设置图表标签
        plt.xlabel('Value Range')
        plt.ylabel('Frequency')
        plt.title('')
        plt.savefig(save_name)

    
    def show_result(self, conf_res, tp_res):
        tp_th = [item[0]['th'] for item in tp_res]
        tp_true_prob = [item[0]['true_prob'] for item in tp_res]
        tp_info_gain = [item[0]['info_gain'] for item in tp_res]

        conf_th = [item[0]['th'] for item in conf_res]
        conf_true_prob = [item[0]['true_prob'] for item in conf_res]
        conf_info_gain = [item[0]['info_gain'] for item in conf_res]

        # 创建图形
        fig, ax1 = plt.subplots()

        # 设置左边的纵轴
        ax1.set_xlabel('Threshold (th)')
        ax1.set_ylabel('True Probability', color='tab:blue')
        ax1.plot(tp_th, tp_true_prob, label='TP True Probability', color='tab:blue', marker='o')
        ax1.plot(conf_th, conf_true_prob, label='Conf True Probability', color='tab:orange', marker='o')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')

        # 创建右边的纵轴
        ax2 = ax1.twinx()
        ax2.set_ylabel('Information Gain', color='tab:red')
        ax2.plot(tp_th, tp_info_gain, label='TP Info Gain', color='tab:red', linestyle='--', marker='x')
        ax2.plot(conf_th, conf_info_gain, label='Conf Info Gain', color='tab:green', linestyle='--', marker='x')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='upper right')

        # 显示图形
        plt.title('True Probability and Information Gain vs. Threshold')
        plt.grid(False)
        plt.savefig('case_study3.png')
        plt.show()
    
    def __call__(self):
        conf_res = self.get_conf_range_info()
        tp_res = self.get_tp_range_info()
        self.show_result(conf_res, tp_res)


class CaseStudy2(InfoGainCaculater):
    def __init__(self, args, data_type='train_lb'):
        super().__init__(args, data_type)
        self.args = args
        self.topk_num = 2000
        self.save_dir = '/saved_models/two_phase/visual/cifar_400'
        
        self.conf_ckpt_list = self.sort_path(glob(osp.join(self.save_dir, 'plot2pseudo', 'model_*_*.pth')))
        self.conf_td_last = torch.load(osp.join(self.save_dir, 'plot2pseudo', 'tp_model_574_26.pth'), map_location='cpu')
        
        self.tp_ckpt_list =  self.sort_path(glob(osp.join(self.save_dir, 'plot2tp', 'model_*_*.pth')))
        self.tp_td_init = torch.load('/saved_models/two_phase/supervised/cifar_400/record_epoch2/tp_latest_model.pth', map_location='cpu')
        self.tp_td_last = torch.load(osp.join(self.save_dir, 'plot2tp', 'tp_model_574_26.pth'), map_location='cpu')

        self.y = torch.from_numpy(self.data_loader.dataset.targets).long()

    @ staticmethod
    def sort_path(file_paths):
        # 获取文件的原始顺序
        original_order = {f: os.stat(f).st_mtime for f in file_paths}
        # 按照原始顺序排序
        sorted_files = sorted(file_paths, key=lambda x: original_order[x])
        return sorted_files
    
    def build_subset(self, idxs):
        x, y = [], []
        for i in idxs:
           data_warrper =  self.data_loader.dataset.__getitem__(0)
           x.append(data_warrper['x_ulb_w'])
           y.append(data_warrper['y_ulb'])

        x = torch.stack(x)
        y = torch.tensor(y).long()
        
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        return x, y
    
    def get_info_gain_gap(self, subset_gain, y_hat, top_idx):
        true_mask = (y_hat[top_idx] == self.y[top_idx])
        true_num = true_mask.sum()
        true_prob = true_num/true_mask.shape[0]
        
        if true_mask.sum() == 0: # 全错
            true_gain = torch.tensor(0)
            false_gain = subset_gain[~true_mask].sum()
        elif true_mask.sum() == true_mask.shape[0]: # 全对
            true_gain = subset_gain[true_mask].sum()
            false_gain = torch.tensor(0)
        else:
            true_gain = subset_gain[true_mask].sum()
            false_gain = subset_gain[~true_mask].sum()
            
        gain_gap = true_gain - false_gain
        return true_num, true_prob, gain_gap, true_gain, false_gain


class CaseStudy2v2(CaseStudy2):
    def __init__(self, args, data_type='train_lb'):
        super().__init__(args, data_type)
        self.args = args
        self.topk_num = 2000
        self.data_type = data_type

        self.save_dir = '/saved_models/two_phase/ups/'
        self.num_classess = {'cifar': 100, 'stl10': 10,'euro': 10, 'semi': 200}
        self.data_name = {'cifar': 'cifar100', 'euro': 'eurosat', 'semi': 'semi_aves', 'stl10':'stl10'}
        self.ckpt_list = glob(self.save_dir+'*/record_epoch4/latest_model.pth')
        self.tp_record_path_list = glob(self.save_dir+'*/record_epoch4/tp_latest_model.pth')

    def single_conf(self, tp_record):
        logit_now = tp_record['training_dynamic'][:,:,-1]
        confidence, y_hat = logit_now.max(-1)

        _, top_idx = confidence.topk(self.topk_num)
        x, y = self.build_subset(top_idx)
        subset_gain = self.get_subset_gain(x, y)

        true_num, true_prob, gain_gap, true_gain, false_gain = self.get_info_gain_gap(subset_gain, y_hat, top_idx)
        return  {
            'true_num': true_num.item(), 
            'true_prob': true_prob.item(), 'gain_gap': gain_gap.item(),
            'true_gain': true_gain.item(), 'false_gain': false_gain.item()
        }
    
    def single_tp(self, tp_record):
        _, y_hat = tp_record['training_dynamic'][:,:,-1].max(-1)
        pre_entropy = caculate_tp(self.args, tp_record['training_dynamic'], tp_record['delta_softmax'], y_hat)

        _, top_idx = (-pre_entropy).topk(self.topk_num)
        x, y = self.build_subset(top_idx)
        subset_gain = self.get_subset_gain(x, y)
        true_num, true_prob, gain_gap, true_gain, false_gain = self.get_info_gain_gap(subset_gain, y_hat, top_idx)
        return  {
            'true_num': true_num.item(), 
                'true_prob': true_prob.item(), 'gain_gap': gain_gap.item(),
                'true_gain': true_gain.item(), 'false_gain': false_gain.item()
                }
    
    def get_all_init_gain(self):
        all_res = []
        for ckpt_path, tp_path in zip(self.ckpt_list, self.tp_record_path_list):
            data_key = ckpt_path.split('/')[-3]
            args.dataset = self.data_name[data_key.split('_')[0]]
            args.num_labels = int(data_key.split('_')[-1])
            args.num_classes = self.num_classess[data_key.split('_')[0]]

            print(args.dataset)
            if args.dataset in ['stl10', 'semi_aves']:
                continue
            self.net = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes)
            self.load_model(ckpt_path)
            
            dataset_dict = get_dataset(args, 'pseudolabel', args.dataset, args.num_labels, args.num_classes, args.data_dir, True)
            dataset = dataset_dict[self.data_type]
            self.data_loader = DataLoader(dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=4)
            self.y = torch.from_numpy(self.data_loader.dataset.targets).long()
            
            tp_record = torch.load(tp_path, map_location='cpu')

            conf_res = self.single_conf(tp_record)
            conf_res['title'] = data_key+'_conf'
            print(conf_res)
            all_res.append(conf_res)

            tp_res = self.single_tp(tp_record)
            tp_res['title'] = data_key+'_tp'
            print(tp_res)
            all_res.append(tp_res)

        with open('casestudy2.json', 'w') as f:
            json.dump(all_res, f, ensure_ascii=False, indent=4)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, default='saved_models/two_phase/visual/cifar_400/batch8epoch5/model_24979_21.pth')
    parser.add_argument('--tp_load_path', type=str, default='saved_models/two_phase/visual/cifar_400/batch8epoch5/tp_model_24979_21.pth')
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='vit_small_patch2_32')
    parser.add_argument('--net_from_name', type=bool, default=False)

    parser.add_argument('--alpha', type=float, default=0.4468)
    parser.add_argument('--beta', type=float, default=0.8668)
    parser.add_argument('--eta1', type=float, default=0.64975)
    parser.add_argument('--eta2', type=float, default=0.645306)
    parser.add_argument('--eta3', type=float, default=0.468570)
    parser.add_argument('--b1', type=float, default=0.05289)
    parser.add_argument('--b2', type=float, default=0.1926027)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--data_dir', type=str, default='/data')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=int, default=0.875)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)

    args = parser.parse_args()

    args.num_labels = 400
    args.ulb_num_labels = None
    args.lb_imb_ratio = 1
    args.ulb_imb_ratio = 1
    args.seed = 0
    args.epoch = 1
    args.num_train_iter = 1024
    args.is_mini_data = False

    # reproduce tabel 4
    caculater = CaseStudy2v2(args, 'train_lb')
    caculater.get_all_init_gain()

    # reproduce fig 6
    caculater1 = CaseStudy3(args, 'train_ulb')
    caculater1()