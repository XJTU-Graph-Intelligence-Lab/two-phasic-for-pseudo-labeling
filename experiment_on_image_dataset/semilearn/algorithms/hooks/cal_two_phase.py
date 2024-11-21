import torch
import torch.nn.functional as F
from semilearn.core.hooks import Hook
import numpy as np
import os.path as osp

def entropy(p):
    log = torch.log(p)
    log = torch.where(torch.isinf(log), torch.full_like(log, 0), log)
    entropy = torch.sum(torch.mul(-p, log), dim=1)
    return entropy

class CAL_TP(Hook):
    def __init__(self):
        super().__init__()

    def get_mask_from_origin(self, algorithm, idx_ulb):
        mask = algorithm.origin_alg_mask[idx_ulb]
        pseudo_label = algorithm.origin_alg_y[idx_ulb]
        return mask, pseudo_label
    
    @torch.no_grad()
    def record_td(self, algorithm, logits_all):
        # 记录每个batch的训练动态
        # idx_ulb:当前batch所有未标注数据的idx
        # y_ulb:当前batch所有未标注数据的标签
        pre = F.softmax(logits_all.cuda(algorithm.gpu), dim=1)
        algorithm.training_dynamic[:, :, algorithm.record_id] = pre

        if algorithm.record_id == 0:
            algorithm.last_pre = pre
        else:
            algorithm.delta_softmax  += torch.absolute(pre - algorithm.last_pre)
            algorithm.last_pre = pre
        
        algorithm.sum_softmax += pre

    @ torch.no_grad()
    def calculate_tp(self, algorithm):
        # 计算全局的tp伪标签
        tp_idx = []
        # 训练集|被原方法选中的
        non_zero_mask = (algorithm.training_dynamic.sum(0).sum(0)!=0)
        training_dynamic = algorithm.training_dynamic[:,:,non_zero_mask]

        count = training_dynamic.shape[-1]
        
        # 空间
        sum_softmax = (training_dynamic.sum(-1) / count).cuda(algorithm.gpu)
        # train_mask = algorithm.origin_alg_mask.bool()
        sum_entropy = entropy(sum_softmax).cuda(algorithm.gpu)

        # 变化稳定
        delta_tp, top2_idx = torch.topk(sum_softmax, 2, dim=-1)
        
        # 初始化twin_peak_index，第一列为 pre_labels
        twin_peak_index = torch.zeros_like(top2_idx)
        twin_peak_index[:, 0] = algorithm.pre_label
        # 创建布尔掩码，检查 pre_labels 是否在 top2_idx 中
        is_first = (algorithm.pre_label.unsqueeze(1).tile(2) == top2_idx)
        # 选择第二列的值, is_first[:, 0] == True代表pre和top2_idx第一列一致，选第二列，反之选第一列
        twin_peak_index[:, 1] = torch.where(is_first[:, 0], 
                                      top2_idx[:, 1], 
                                      top2_idx[:, 0]) 
        # 处理未匹配的情况, 选top1
        twin_peak_index[~is_first.any(dim=1), 1] = top2_idx[~is_first.any(dim=1), 0]

        delta_not_tp = algorithm.delta_softmax.sum(dim=1) - torch.gather(algorithm.delta_softmax, 1, top2_idx).sum(1)
        time_twin_peak = delta_tp.sum(1) / 2
        time_not_twin_peak = delta_not_tp / (algorithm.args.num_classes-2)
        k_twin_peak = 1 / (1 + torch.exp(-(time_twin_peak - algorithm.args.b1)))
        k_not_twin_peak = 1 / (1 + torch.exp(-(time_not_twin_peak - algorithm.args.b2)))
        change_stable = (k_twin_peak)**algorithm.args.alpha * (k_not_twin_peak)**algorithm.args.beta

        # 变化存在
        g_final_tmp = torch.gather(training_dynamic[:,:,-1], 1, twin_peak_index.cpu())
        g_final = g_final_tmp[:, 0] - g_final_tmp[:, 1]
        g_min_tmp = torch.gather(training_dynamic, 1, twin_peak_index.unsqueeze(-1).repeat(1, 1, training_dynamic.shape[-1]).cpu())
        g_min = (g_min_tmp[:, 0] - g_min_tmp[:, 1]).min(dim=-1)[0]
        change_exist = (g_final - g_min).cuda(algorithm.gpu) + 1e-3
        ## 形成指标
        pre_entropy = sum_entropy**algorithm.args.eta1 * change_stable**algorithm.args.eta2 * (1/change_exist)**algorithm.args.eta3
        
        #selecting positive pseudo-labels
        ## 用two phase筛选
        # self.plot_in_intro(algorithm, training_dynamic, pre_entropy)
        algorithm.args.tp_thres = pre_entropy.mean()
        for i in range(algorithm.args.num_classes):
            entropy_index = torch.where(algorithm.pre_label==i)[0]  # 选伪标签排除训练集
            v, ind = pre_entropy[entropy_index].sort()
            if entropy_index.size(0) > 0: # 如果有伪标签
                new_modis_num = sum(v <= algorithm.args.tp_thres).item()
                tp_idx += entropy_index[ind][: new_modis_num].cpu().tolist() # 小于阈值的全选
        tp_num = len(tp_idx)

        negetive_mask = torch.zeros_like(sum_softmax).to(algorithm.gpu)
        negetive_mask[tp_idx, top2_idx[:,1][tp_idx]] = 1
        negetive_mask = negetive_mask.bool()
        negetive_idx = torch.where(negetive_mask==1)[0]
        negetive_num = negetive_idx.shape[0]

        if tp_num > 0:
            tp_pseudo_acc = (algorithm.pre_label[tp_idx] == algorithm.real_label[tp_idx]).sum().item() / tp_num
        else:
            tp_pseudo_acc = 0
        algorithm.tp_pseudo_dict = {"tp_idx" : tp_idx, "neg_idx" : negetive_idx, "neg_mask" : negetive_mask,
                          "tp_num" : tp_num, "negetive_num" : negetive_num, "tp_pseudo_acc" : tp_pseudo_acc}

    def calculate_loss(self, algorithm, idx_ulb, mask, logits_x_ulb_w):
        if algorithm.epoch > algorithm.args.tp_epoch-1 and algorithm.args.use_tp:
                neg_idx = list(set(algorithm.tp_pseudo_dict["neg_idx"].cpu().tolist()) & set(idx_ulb.cpu().tolist()))
                neg_mask = algorithm.tp_pseudo_dict["neg_mask"][neg_idx]
                neg_idx = [idx_ulb.cpu().tolist().index(item) for item in neg_idx]
                tp_num = algorithm.tp_pseudo_dict["tp_num"]
                negetive_num = algorithm.tp_pseudo_dict["negetive_num"]
                tp_pseudo_acc = algorithm.tp_pseudo_dict.get("tp_pseudo_acc", 0)

                tp_pseudo_idx = list(set(algorithm.tp_pseudo_dict["tp_idx"]) & set(idx_ulb.cpu().tolist()))
                tp_pseudo_label = algorithm.pre_label[tp_pseudo_idx] 
                intersection_indices = [idx_ulb.cpu().tolist().index(item) for item in tp_pseudo_idx]
                if len(intersection_indices) > 0:  # 防止没有tp样本
                    tp_loss = algorithm.tp_pseudo_loss(logits_x_ulb_w[intersection_indices], tp_pseudo_label, reduction='mean')
                else:
                    tp_loss = 0  
                
                if neg_mask is not None and len(neg_idx) > 0: ## 防止没有负样本
                    nl_logits = logits_x_ulb_w[neg_idx]
                    pred_nl = F.softmax(nl_logits, dim=1)
                    pred_nl = 1 - pred_nl
                    pred_nl = torch.clamp(pred_nl, 1e-7, 1.0)
                    nl_mask = neg_mask
                    y_nl = torch.ones((nl_logits.shape)).to(device=nl_mask.device, dtype=logits_x_ulb_w.dtype)
                    loss_nl = torch.mean((-torch.sum((y_nl * torch.log(pred_nl))*nl_mask, dim = -1))/(torch.sum(nl_mask, dim = -1) + 1e-7))
                else:
                    loss_nl = 0
        else:
            neg_idx = None
            neg_mask = None
            tp_num = 0
            negetive_num = 0
            tp_pseudo_acc = 0
            tp_loss = 0 
            loss_nl = 0
        return neg_idx, neg_mask, tp_num, negetive_num, tp_pseudo_acc, tp_loss, loss_nl
    
    @ torch.no_grad()
    def modis_get_pseudo(self, algorithm):
        # 计算全局的modis伪标签
        modis_idx = []

        training_dynamic = algorithm.training_dynamic
        pre_label = algorithm.pre_label
        count = training_dynamic.shape[-1]
        training_dynamic = training_dynamic ** (1/algorithm.args.modis_T)
        training_dynamic = training_dynamic / training_dynamic.sum(dim=1, keepdim=True)
        sum_softmax = training_dynamic.sum(-1) / count
        pre_entropy = entropy(sum_softmax).cuda(algorithm.gpu)
        print(pre_entropy.min())

        algorithm.args.modis_thres = pre_entropy.mean()
        ## 用modis筛选
        for i in range(algorithm.args.num_classes):
            entropy_index = torch.where(pre_label==i)[0]
            v, ind = pre_entropy[entropy_index].sort()
            if entropy_index.size(0) > 0: # 如果有伪标签
                new_modis_num = sum(v <= algorithm.args.modis_thres).item()
            modis_idx += entropy_index[ind][: new_modis_num].cpu().tolist() # 小于阈值的全选
        modis_idx_num = len(modis_idx)

        if modis_idx_num > 0:
            modis_pseudo_acc = (algorithm.pre_label[modis_idx] == algorithm.real_label[modis_idx]).sum().item() / modis_idx_num
        else:
            modis_pseudo_acc = 0
        print(f'modis_idx_num: {modis_idx_num}, modis_pseudo_acc: {modis_pseudo_acc}')
        algorithm.modis_pseudo_dict = {"modis_idx" : modis_idx, "modis_idx_num" : modis_idx_num, "modis_pseudo_acc" : modis_pseudo_acc,
                                       "pre_label" : pre_label}

    @staticmethod
    def find_conf_threshold(probabilities, labels,
                       accuracy_threshold=0.75, start=0.5, end=0.99, gap=0.05):
        # 获取预测的类
        predicted_labels = np.argmax(probabilities, axis=1)

        # 遍历置信度阈值
        for confidence_threshold in np.arange(start, end, gap):
            # 计算当前准确率
            high_confidence_indices = np.where(np.max(probabilities, axis=1) >= confidence_threshold)[0]
            correct_predictions = (predicted_labels[high_confidence_indices] == labels[high_confidence_indices])
            accuracy = np.mean(correct_predictions)

            # 检查是否达到准确率阈值
            if accuracy >= accuracy_threshold:
                return confidence_threshold, high_confidence_indices.shape[0], high_confidence_indices  # 返回满足条件的置信度阈值

        return "none", 0, []  # 如果没有找到合适的阈值，返回 None

    @ staticmethod
    def find_tp_threshold(pre_entropy, pre_label, real_label, accuracy_threshold=0.75):
        start, end = pre_entropy.mean()-0.3, pre_entropy.mean()+0.3
        all_res = []
        for tp_thres in np.arange(start.item(), end.item(), 0.01):
            tp_idx = pre_entropy < tp_thres
            acc = (pre_label[tp_idx] == real_label[tp_idx]).sum()/tp_idx.sum()
            if acc >= accuracy_threshold:
                all_res.append(
                   {'tp_thres': tp_thres, 'tp_num': tp_idx.sum(), 'tp_idx': tp_idx.cpu().numpy()}
                ) 
        if all_res:
            return all_res[-1]
        else:
            return {}

    def plot_in_intro(self, algorithm, training_dynamic, pre_entropy, accuracy_threshold=0.75):
        leagal_res = []
        for i in range(training_dynamic.shape[-1]):
            logits = training_dynamic[:,:,i].cpu().numpy()
            p, n, conf_idx = self.find_conf_threshold(logits, algorithm.real_label.cpu().numpy(), accuracy_threshold=accuracy_threshold)
            if p != "none":
                leagal_res.append(
                    {'td_idx': i, 'conf_idx': conf_idx}
                )
                print(f"td_idx: {i}, conf_num: {n}")
        tp_res = self.find_tp_threshold(pre_entropy, algorithm.pre_label, algorithm.real_label, accuracy_threshold)
        
        if tp_res == {}:
            return "please input lowwer accuracy_threshold"
        else:
            tp_thres, tp_num, tp_idx = tp_res['tp_thres'], tp_res['tp_num'], tp_res['tp_idx']
            print(f"{tp_thres} get {tp_num} tp sample")
        
        # 计算交集个数
        num_list = []
        for res in leagal_res:
           num_list.append(
               np.intersect1d(res['conf_idx'], np.where(tp_idx)[0]).shape[0]
           )
        print('please select!')
        return num_list, leagal_res, tp_num, tp_idx
