import argparse
import json
import torch as th
import torch.nn.functional as F
import numpy as np
import utils_data
import collections
import time
import optuna
from utils import get_models, seed_everything
import os.path as osp
path_root = osp.abspath(osp.dirname(__file__))

def train(model, graph, features, train_labels, labels, train_mask, val_mask, optimizer, neg_mask):
    model.train()
    train_logits = model(graph, features)
    train_logp = F.log_softmax(train_logits, 1)
    if neg_mask is None:
        train_loss = F.nll_loss(train_logp[train_mask], train_labels[train_mask])
    else:
        train_loss = F.nll_loss(train_logp[train_mask], train_labels[train_mask])
        ## c1 loss
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
    ## choose two-phase for every class
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

def main_train(args, trial):
    args.run_id = args.run_id.replace('\r', '')

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    seed_range = [1111]
    test_acc = th.zeros(len(seed_range))
    tp_pse_acc = th.zeros(len(seed_range))
    tp_pseudo_num = []

    time_now = time.time()
    for i, seed in enumerate(seed_range):
        seed_everything(seed)
        g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, adj = \
            utils_data.load_data(args.dataset, args.train_size, args.val_size, device)
        
        net = get_models(args, num_features, num_labels)
        if args.model == 'GCNII':
            optimizer = th.optim.Adam([
                {'params': net.params1, 'weight_decay': args.wd1},
                {'params': net.params2, 'weight_decay': args.wd2},
            ], lr=args.learning_rate)
        else:
            optimizer = th.optim.Adam([{'params': net.parameters(), 'weight_decay': args.weight_decay}],
                                      lr=args.learning_rate)

        net.to(device)
        features = features.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        all_tp_pseudo = []

        ### load pseudo labels of baseline method
        train_mask_running = th.load(osp.join(path_root, 'origin_pseudo', f'{args.name}', 
                                              f'train_mask_running_{args.train_size}_{args.dataset}_{seed}.pth')).to(device)
        pre_labels = th.load(osp.join(path_root, 'origin_pseudo', f'{args.name}', 
                                      f'pre_labels_{args.train_size}_{args.dataset}_{seed}.pth')).to(device)

        print("------------------- PreTrain -------------------")
        for stage in range(args.stage):
            sum_softmax = th.zeros((labels.shape[0], num_labels)).to(device)
            delta_softmax = th.zeros((labels.shape[0], num_labels)).to(device)
            last_pre = th.zeros((labels.shape[0], num_labels)).to(device)
            training_dynamics = th.zeros((labels.shape[0], num_labels, args.num_epochs_max)).to(device)
            neg_mask = None

            net.reset_parameters()
            net.to(device)
            optimizer.state = collections.defaultdict(dict)
            print("------------------- Stage {:02d} -------------------".format(stage))
            for epoch in range(args.num_epochs_max):
                val_acc, val_loss = train(net, g, features, pre_labels, labels, train_mask_running, val_mask, optimizer, neg_mask=neg_mask)
                # print("Epoch {:05d} | Val Loss {:.4f} | Val Acc {:.4f}".format(epoch, val_loss, val_acc))
                net.eval()
                with th.no_grad():
                    logits = net(g, features)
                    pre = F.softmax(logits, 1)
                    training_dynamics[:, :, epoch] = pre
                if epoch == 0:
                    last_pre = pre
                else:
                    delta_softmax += th.absolute(pre - last_pre)
                    last_pre = pre
                sum_softmax += pre
            ## Preventing prediction errors in the training dataset
            pre_labels = pre.argmax(dim=1)
            pre_labels[train_mask] = labels[train_mask]
            ## Spatial Measure
            sum_softmax = sum_softmax / args.num_epochs_max
            max_v, _ = th.max(sum_softmax, dim=1, keepdim=True)
            sum_softmax = th.where(sum_softmax==max_v, 0, sum_softmax)
            sum_entropy = entropy(sum_softmax)
            ## c1 and c2 label
            _, twin_peak_index = th.topk(sum_softmax, k=2)
            pre_not_in_tp_index = th.where((pre_labels.unsqueeze(1).tile(2)==twin_peak_index).sum(1)==0)[0]
            c2 = twin_peak_index[~(pre_labels.unsqueeze(1).tile(2)==twin_peak_index)]
            c2[pre_not_in_tp_index + 1] = -1
            c2 = c2[c2 > -1]
            c1_mask = th.zeros_like(sum_softmax).bool()
            c1_mask[range(labels.shape[0]), pre_labels] = True
            c2_mask = th.zeros_like(sum_softmax).bool()
            c2_mask[range(labels.shape[0]), c2] = True
            ## Observation 1
            delta_tp = th.gather(delta_softmax, 1, twin_peak_index)
            delta_not_tp = delta_softmax.sum(dim=1) - delta_tp.sum(dim=1)
            time_twin_peak = delta_tp.sum(1) / 2
            time_not_twin_peak = delta_not_tp / (num_labels-2)
            k_twin_peak = 1 / (1 + th.exp(-(time_twin_peak - args.b1)))
            k_not_twin_peak = 1 / (1 + th.exp(-(time_not_twin_peak - args.b2)))
            change_stable = (k_twin_peak)**args.alpha * (k_not_twin_peak)**args.beta
            ## Observation 2
            g_final = training_dynamics[:,:,-1][c1_mask] - training_dynamics[:,:,-1][c2_mask]
            g_min = (training_dynamics[c1_mask] - training_dynamics[c2_mask]).min(dim=1)[0]
            change_exist = g_final - g_min
            ## 2-phasic metric
            pre_entropy = sum_entropy**args.eta1 * change_stable**args.eta2 * (1/change_exist)**args.eta3

            train_mask_running, tp_pseudo = selftraining(pre_labels, pre_entropy,
                                              num_labels, train_mask_running, test_mask, device)
            print('2-phase num:', len(tp_pseudo))
            print('2-phase acc:',th.eq(pre_labels[tp_pseudo], labels[tp_pseudo]).float().mean().item() * 100.0)
            all_tp_pseudo += tp_pseudo
            neg_mask = th.zeros_like(sum_softmax).to(device)
            
            neg_mask[all_tp_pseudo, c2[all_tp_pseudo]] = 1
            neg_mask = neg_mask.bool()

        print("------------------- Last Train -------------------")
        net.reset_parameters()
        net.to(device)
        optimizer.state = collections.defaultdict(dict)
        vlss_mn = np.inf
        vacc_mx = 0.0
        state_dict_early_model = None
        for epoch in range(args.num_epochs_max):
            val_acc, val_loss = train(net, g, features, pre_labels, labels, train_mask_running, val_mask, optimizer, neg_mask=neg_mask)
            # print("Epoch {:05d} | Val Loss {:.4f} | Val Acc {:.4f}".format(epoch, val_loss, val_acc))
            if val_acc >= vacc_mx or val_loss <= vlss_mn:
                if val_acc >= vacc_mx and val_loss <= vlss_mn:
                    state_dict_early_model = net.state_dict()
                vacc_mx = np.max((val_acc, vacc_mx))
                vlss_mn = np.min((val_loss, vlss_mn))

        net.load_state_dict(state_dict_early_model)
        net.eval()
        with th.no_grad():
            test_logits = net(g, features)
            test_logp = F.log_softmax(test_logits, 1)
            test_loss = F.nll_loss(test_logp[test_mask], labels[test_mask]).item()
            test_pred = test_logp.argmax(dim=1)
            test_acc[i] = th.eq(test_pred[test_mask], labels[test_mask]).float().mean().item() * 100.0
            tp_pse_acc[i] = th.eq(pre_labels[all_tp_pseudo], labels[all_tp_pseudo]).float().mean().item() * 100.0
            tp_pseudo_num.append(len(all_tp_pseudo))
        print(test_acc[i].item())
    results_dict = vars(args)
    results_dict['test_loss'] = test_loss
    results_dict['test_acc_mean'] = test_acc.mean().item()
    results_dict['test_acc_std'] = test_acc.std(dim=0).item()
    results_dict['tp_pseudo_acc_mean'] = tp_pse_acc[~th.isnan(tp_pse_acc)].mean().item()
    results_dict['tp_pseudo_num'] = tp_pseudo_num
    results_dict['total_time'] = time.time() - time_now
    with open(osp.join(path_root, 'final_runs', f'{args.name}_{args.dataset}_{args.train_size}_'
                                   f'{args.run_id}_results.json'), 'a') as outfile:
        json.dump(results_dict, outfile, indent=4)
    return results_dict['test_acc_mean']

def set_search_space(trial):
    args.alpha = trial.suggest_float('alpha', 0, 0.5)
    args.beta = trial.suggest_float('beta', 0.5, 1)
    args.eta1 = trial.suggest_float('eta1', 0, 1)
    args.eta2 = trial.suggest_float('eta2', 0, 1)
    args.eta3 = trial.suggest_float('eta3', 0, 1)
    args.b1 = trial.suggest_float('b1', 0, 0.1)
    args.b2 = trial.suggest_float('b2', 0.1, 0.2)
    args.tp_threshold = trial.suggest_float('tp_threshold', 0.8, 1.8)
    if args.dataset == 'AmazonCoBuyComputer':
        args.lr = trial.suggest_float('lr', 1e-3, 1e-2)
    else:
        args.lr = trial.suggest_float('lr', 0.01, 0.1)
    test_acc = main_train(args, trial)
    return test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--name', type=str, default='modis') # modis aum st
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.01)
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
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--eta1', type=float, default=1)
    parser.add_argument('--eta2', type=float, default=1)
    parser.add_argument('--eta3', type=float, default=1)
    parser.add_argument('--b1', type=float, default=0.01)
    parser.add_argument('--b2', type=float, default=0.15)
    parser.add_argument('--tp_threshold', type=float, default=1.3)
    parser.add_argument('--run_id', type=str, default='0')

    args = parser.parse_args()
    # test_acc = main_train(args, None)
    
    ## Parameter tuning
    study = optuna.create_study(direction='maximize')
    study.optimize(set_search_space, n_trials=20)
    beat_res = study.best_params
    beat_res['value'] = study.best_value
    with open(osp.join(path_root, 'final_runs', f'{args.name}', f'{args.name}_{args.dataset}_{args.train_size}_{args.run_id}_results1.json'), 'a') as outfile:
        json.dump(beat_res, outfile, indent=4)
    for trial in study.trials:
        res = {}
        res['number'] = trial.number    
        res['params'] = trial.params
        res['value'] = trial.value
        with open(osp.join(path_root, 'final_runs', f'{args.name}', f'{args.name}_{args.dataset}_{args.train_size}_{args.run_id}_results1.json'), 'a') as outfile:
            json.dump(res, outfile, indent=4)