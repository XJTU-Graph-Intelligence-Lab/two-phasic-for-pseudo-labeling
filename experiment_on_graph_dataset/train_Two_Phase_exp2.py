import argparse
import torch as th
import torch.nn.functional as F
import numpy as np
import utils_data
from utils import get_models, seed_everything
import os.path as osp
path_root = osp.abspath(osp.dirname(__file__))

from tqdm import tqdm


class experiment2():
    def __init__(self, y, confi_metric, tp_metric, pre_labels, all_gain):
        self.y = y
        self.confi_metric = confi_metric
        self.all_info_gain = all_gain
        self.tp_metric = tp_metric
        self.y_pre = pre_labels

    def norm_metric(self, tensor):
        tensor_min = tensor.min()
        tensor_max = tensor.max()

        if tensor_max - tensor_min == 0:
            return th.zeros_like(tensor)
        
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        return normalized_tensor
    
    def get_conf_range_info(self):
        confidence = self.confi_metric
        y_hat_c = self.y_pre
        norm_conf = self.norm_metric(confidence)

        conf_res = []
        for conf_th in tqdm(np.arange(50, 750, 50)):
            _, indices = th.topk(confidence, conf_th)

            # Create a mask for the top k samples
            pseudo_mask = th.zeros_like(norm_conf, dtype=bool)
            pseudo_mask[indices] = True
            # pseudo_mask = norm_conf>conf_th

            true_mask = y_hat_c == self.y
            info_gain_all = self.all_info_gain[true_mask].sum() - self.all_info_gain[~true_mask].sum()

            true_num = (y_hat_c[pseudo_mask] == self.y[pseudo_mask]).sum()
            true_prob = true_num/pseudo_mask.sum()
            true_gain = self.all_info_gain[th.logical_and(true_mask, pseudo_mask)].sum()
            flase_gain = self.all_info_gain[th.logical_and(~true_mask, pseudo_mask)].sum()
            info_gain =  (10*true_gain - flase_gain)

            conf_res.append([
                {
                   'th':conf_th, 'true_num': true_num.item(), 'true_prob': true_prob.item(), 
                   'info_gain': info_gain.item(), 'true_gain': true_gain.item(), 'flase_gain': flase_gain.item()
                }
            ])
        return conf_res
    
    def get_tp_range_info(self):
        y_hat = self.y_pre
        norm_tp = self.norm_metric(-F.sigmoid(self.tp_metric)) 

        tp_res = []
        for tp_th in tqdm(np.arange(50, 750, 50)):
            _, indices = th.topk(norm_tp, tp_th)
    
            # Create a mask for the top k samples
            pseudo_mask = th.zeros_like(norm_tp, dtype=bool)
            pseudo_mask[indices] = True
            # pseudo_mask = norm_tp>tp_th
            true_mask = y_hat == self.y
            info_gain_all = self.all_info_gain[true_mask].sum() - self.all_info_gain[~true_mask].sum()

            true_num = (y_hat[pseudo_mask] == self.y[pseudo_mask]).sum()
            true_prob = true_num/pseudo_mask.sum()
            true_gain = self.all_info_gain[th.logical_and(true_mask, pseudo_mask)].sum()
            flase_gain = self.all_info_gain[th.logical_and(~true_mask, pseudo_mask)].sum()
            info_gain = (10*true_gain - flase_gain)

            tp_res.append([
                {
                    'th':tp_th, 'true_num': true_num.item(), 'true_prob': true_prob.item(), 
                    'info_gain': info_gain.item(), 'true_gain': true_gain.item(), 'flase_gain': flase_gain.item()
                }
            ])
        return tp_res
    
    def __call__(self):
        conf_res = self.get_conf_range_info()
        tp_res = self.get_tp_range_info()
        return conf_res, tp_res


def train(model, graph, features, train_labels, labels, train_mask, val_mask, optimizer, neg_mask):
    model.train()
    train_logits = model(graph, features)
    train_logp = F.log_softmax(train_logits, 1)
    if neg_mask is None:
        train_loss = F.nll_loss(train_logp[train_mask], train_labels[train_mask])
    else:
        train_loss = F.nll_loss(train_logp[train_mask], train_labels[train_mask])
        neg_idx = th.where(neg_mask.sum(dim=1)!=0)[0]
        neg_logits = train_logits[neg_idx]
        neg_loss = F.softmax(neg_logits, 1)
        neg_loss = 1 - neg_loss
        neg_loss = th.clamp(neg_loss, 1e-7, 1.0)
        y_nl = th.ones((neg_logits.shape)).to(neg_loss.device)
        loss_nl = th.mean((-th.sum((y_nl * th.log(neg_loss))*neg_mask[neg_idx], dim = -1))/(th.sum(neg_mask[neg_idx], dim = -1) + 1e-7))
        train_loss = train_loss + loss_nl

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with th.no_grad():
        val_logits = model(graph, features)
        val_logp = F.log_softmax(val_logits, 1)
        val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
        val_pred = val_logp.argmax(dim=1)
        val_acc = th.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()
    return val_acc, val_loss


def selftraining(pseudo_labels, entropy, num_class, train_mask, test_mask, device):
    tp_pseudo = []
    n_train_mask = train_mask.clone().to(device)
    for i in range(num_class):
        class_index = th.where(th.logical_and(pseudo_labels == i, ~(train_mask + test_mask)))[0]
        sorted_v, sorted_index = entropy[class_index].sort()
        sorted_index = sorted_index[sorted_v < args.tp_threshold]
        alternate_index = class_index[sorted_index]
        n_train_mask[alternate_index] = True
        tp_pseudo += alternate_index.tolist()
    return n_train_mask, tp_pseudo


def entropy(p):
    log = th.log(p)
    log = th.where(th.isinf(log), th.full_like(log, 0), log)
    entropy = th.sum(th.mul(-p, log), dim=1)
    return entropy


def calgain(model, graph, features, train_labels, right_index, wrong_index):
    if len(right_index) == 0:
        return 0
    model.eval()
    train_logits = model(graph, features)

    gain_right = 0
    for i in right_index:
        train_logp = F.log_softmax(train_logits, 1)
        train_loss = F.nll_loss(train_logp[i], train_labels[i])
        model.zero_grad()
        train_loss.backward(retain_graph=True)
        for _, param in model.named_parameters():
            gain_right += th.norm(param.grad, p='fro')

    gain_wrong = 0
    for i in wrong_index:
        train_logp = F.log_softmax(train_logits, 1)
        train_loss = F.nll_loss(train_logp[i], train_labels[i])
        model.zero_grad()
        train_loss.backward(retain_graph=True)
        for _, param in model.named_parameters():
            gain_wrong += th.norm(param.grad, p='fro')

    return (gain_right - gain_wrong).item()


def choose_pseudo(pre, train_mask, labels, sum_softmax, delta_softmax, num_labels, training_dynamics):
    pre_labels = pre.argmax(dim=1)
    pre_labels[train_mask] = labels[train_mask]

    sum_softmax = sum_softmax / args.num_epochs_max
    max_v, _ = th.max(sum_softmax, dim=1, keepdim=True)
    sum_softmax = th.where(sum_softmax==max_v, 0, sum_softmax)
    sum_entropy = entropy(sum_softmax)

    _, twin_peak_index = th.topk(sum_softmax, k=2)
    pre_not_in_tp_index = th.where((pre_labels.unsqueeze(1).tile(2)==twin_peak_index).sum(1)==0)[0]
    c2 = twin_peak_index[~(pre_labels.unsqueeze(1).tile(2)==twin_peak_index)]
    c2[pre_not_in_tp_index + 1] = -1
    c2 = c2[c2 > -1]
    c1_mask = th.zeros_like(sum_softmax).bool()
    c1_mask[range(labels.shape[0]), pre_labels] = True
    c2_mask = th.zeros_like(sum_softmax).bool()
    c2_mask[range(labels.shape[0]), c2] = True

    delta_tp = th.gather(delta_softmax, 1, twin_peak_index)
    delta_not_tp = delta_softmax.sum(dim=1) - delta_tp.sum(dim=1)
    time_twin_peak = delta_tp.sum(1) / 2
    time_not_twin_peak = delta_not_tp / (num_labels-2)
    k_twin_peak = 1 / (1 + th.exp(-(time_twin_peak - args.b1)))
    k_not_twin_peak = 1 / (1 + th.exp(-(time_not_twin_peak - args.b2)))
    change_stable = (k_twin_peak)**args.alpha * (k_not_twin_peak)**args.beta

    g_final = training_dynamics[:,:,-1][c1_mask] - training_dynamics[:,:,-1][c2_mask]
    g_min = (training_dynamics[c1_mask] - training_dynamics[c2_mask]).min(dim=1)[0]
    change_exist = g_final - g_min

    pre_entropy = sum_entropy**args.eta1 * change_stable**args.eta2 * (1/change_exist)**args.eta3
    max_confidence, _ = th.max(pre, dim=1)
    return pre_entropy, max_confidence


def main_train(args):    
    args.run_id = args.run_id.replace('\r', '')
    seed = 1111
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    seed_everything(seed)
    g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, adj = \
        utils_data.load_data(args.dataset, args.train_size, args.val_size, device)
    
    net = get_models(args, num_features, num_labels)

    net.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    
    net = th.load('model.pth')
    sum_softmax = th.load('sum_softmax.pth')
    delta_softmax = th.load('delta_softmax.pth')
    pre = th.load('pre.pth')
    training_dynamics = th.load('training_dynamics.pth')

    all_gain = []
    for i in range(labels.shape[0]):
        all_gain.append(calgain(net, g, features, labels, [i], []))
    all_gain = th.tensor(all_gain)
    
    pre_labels = pre.argmax(1)
    pre_labels[train_mask] = labels[train_mask]

    tp_metric, confi_metric = choose_pseudo(pre, train_mask, labels, sum_softmax, delta_softmax, num_labels, training_dynamics)
    caculater = experiment2(labels, confi_metric, tp_metric, pre_labels, all_gain)
    conf_res, tp_res = caculater()
    return conf_res, tp_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--name', type=str, default='st') # modis aum st
    parser.add_argument('--dataset', type=str, default='cora') # AmazonCoBuyComputer
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=5e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--wd1', type=float, default=0.01)
    parser.add_argument('--wd2', type=float, default=5e-4)
    parser.add_argument('--bias', type=bool, default=False)
    parser.add_argument('--lamda', type=float, default=0.5)
    parser.add_argument('--variant', action='store_true', default=False)
    parser.add_argument('--T', type=float, default=0.01)
    parser.add_argument('--num_k', type=int, default=112)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--train_size', type=int or float, default=3)
    parser.add_argument('--val_size', type=int or float, default=20)
    parser.add_argument('--num_epochs_patience', type=int, default=100)
    parser.add_argument('--num_epochs_max', type=int, default=400)
    parser.add_argument('--alpha', type=float, default=0.08)
    parser.add_argument('--beta', type=float, default=0.77)
    parser.add_argument('--eta1', type=float, default=0.53)
    parser.add_argument('--eta2', type=float, default=0.77)
    parser.add_argument('--eta3', type=float, default=0.92)
    parser.add_argument('--b1', type=float, default=0.08)
    parser.add_argument('--b2', type=float, default=0.12)
    parser.add_argument('--tp_threshold', type=float, default=1.42)
    parser.add_argument('--run_id', type=str, default='0')

    args = parser.parse_args()
    test_acc = main_train(args)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")