# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import math
import contextlib
import numpy as np
from inspect import signature
from collections import OrderedDict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from semilearn.core.hooks import Hook, get_priority, CheckpointHook, TimerHook, LoggingHook, DistSamplerSeedHook, ParamUpdateHook, EvaluationHook, EMAHook, WANDBHook, AimHook, RecordHook
from semilearn.core.utils import get_dataset, get_data_loader, get_optimizer, get_cosine_schedule_with_warmup, Bn_Controller
from semilearn.core.criterions import CELoss, ConsistencyLoss
from semilearn.algorithms.hooks import CAL_TP


class AlgorithmBase:
    """
        Base class for algorithms
        init algorithm specific parameters and common parameters
        
        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
    """
    def __init__(
        self,
        args,
        net_builder,
        tb_log=None,
        logger=None,
        **kwargs):
        
        # common arguments
        self.args = args
        self.num_classes = args.num_classes
        self.ema_m = args.ema_m
        self.epochs = args.epoch
        self.num_train_iter = args.num_train_iter
        self.num_eval_iter = args.num_eval_iter
        self.num_log_iter = args.num_log_iter
        self.num_iter_per_epoch = int(self.num_train_iter // self.epochs)
        self.lambda_u = args.ulb_loss_ratio 
        self.use_cat = args.use_cat
        self.use_amp = args.amp
        self.clip_grad = args.clip_grad
        self.save_name = args.save_name
        self.save_dir = args.save_dir
        self.resume = args.resume
        self.algorithm = args.algorithm

        # commaon utils arguments
        self.tb_log = tb_log
        self.print_fn = print if logger is None else logger.info
        self.ngpus_per_node = torch.cuda.device_count()
        self.loss_scaler = GradScaler()
        self.amp_cm = autocast if self.use_amp else contextlib.nullcontext
        self.gpu = args.gpu
        self.rank = args.rank
        self.distributed = args.distributed
        self.world_size = args.world_size

        # common model related parameters
        self.it = 0
        self.start_epoch = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.bn_controller = Bn_Controller()
        self.net_builder = net_builder
        self.ema = None

        # build dataset
        self.dataset_dict = self.set_dataset()

        # build data loader
        self.loader_dict = self.set_data_loader()

        self.num_record, self.record_id = 26, 0

        if self.args.new_sample_strategy:
            # 根据epoch, len(dataset), batch_size 设置iter num/num_train_iter
            iterations_per_epoch = math.ceil(len(self.loader_dict['train_ulb'].dataset) / (self.args.batch_size * self.world_size))
            self.num_train_iter = iterations_per_epoch * (self.epochs-self.start_epoch)
            self.num_warmup_iter = int(self.num_train_iter * 0.15)
            
            if self.args.tp_gap == -1:
                self.args.tp_gap = (self.num_train_iter-1)//(self.num_record-1) # 记录126个train dyn

            self.num_eval_iter, self.num_log_iter = iterations_per_epoch//5, iterations_per_epoch//5
            self.print_fn("iterations_per_epoch: {}, num_train_iter: {}".format(iterations_per_epoch, self.num_train_iter))
            self.print_fn(f"tp_gap:{self.args.tp_gap}, num_eval_iter={self.num_eval_iter}, num_log_iter:{self.num_log_iter}")

        # cv, nlp, speech builder different arguments
        self.model = self.set_model()
        self.ema_model = self.set_ema_model()

        # build optimizer and scheduler
        self.optimizer, self.scheduler = self.set_optimizer()

        # build supervised loss and unsupervised loss
        self.ce_loss = CELoss()
        self.consistency_loss = ConsistencyLoss()
        if self.args.use_tp:
            self.tp_pseudo_loss = CELoss()
        else:
            self.tp_pseudo_loss = None

        # other arguments specific to the algorithm
        # self.init(**kwargs)

        # set common hooks during training
        self._hooks = []  # record underlying hooks 
        self.hooks_dict = OrderedDict() # actual object to be used to call hooks
        self.set_hooks()

        # two-phase
        if self.args.use_tp:
            self.sum_softmax = torch.zeros(args.ulb_dest_len, args.num_classes).cuda(self.gpu)
            self.delta_softmax = torch.zeros(args.ulb_dest_len, args.num_classes).cuda(self.gpu)
            self.pre_label = torch.zeros(args.ulb_dest_len, dtype=torch.int64).cuda(self.gpu)
            self.last_pre = torch.zeros(args.ulb_dest_len, args.num_classes).cuda(self.gpu)
            self.real_label = torch.zeros(args.ulb_dest_len, dtype=torch.long).cuda(self.gpu)
            self.training_dynamic = torch.zeros(args.ulb_dest_len, args.num_classes, self.num_record)
        
        self.tp_pseudo_dict = None
        self.origin_alg_mask = torch.zeros(args.ulb_dest_len).cuda(self.gpu)
        if type(self.dataset_dict['train_ulb'].targets) == list:
            self.dataset_dict['train_ulb'].targets = np.array(self.dataset_dict['train_ulb'].targets)
        self.origin_alg_y = torch.from_numpy(self.dataset_dict['train_ulb'].targets).long().cuda(self.gpu)

    def init(self, **kwargs):
        """
        algorithm specific init function, to add parameters into class
        """
        raise NotImplementedError
    

    def set_dataset(self):
        """
        set dataset_dict
        """
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        dataset_dict = get_dataset(self.args, self.algorithm, self.args.dataset, self.args.num_labels, self.args.num_classes, self.args.data_dir, self.args.include_lb_to_ulb)
        if dataset_dict is None:
            return dataset_dict

        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        self.print_fn("unlabeled data number: {}, labeled data number {}".format(self.args.ulb_dest_len, self.args.lb_dest_len))
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict

    def set_data_loader(self):
        """
        set loader_dict
        """
        if self.dataset_dict is None:
            return
            
        self.print_fn("Create train and test data loaders")
        loader_dict = {}
        if self.args.new_sample_strategy:
            sample_num = len(self.dataset_dict['train_ulb'])
        else:
            None

        loader_dict['train_lb'] = get_data_loader(self.args,
                                                  self.dataset_dict['train_lb'],
                                                  self.args.batch_size,
                                                  data_sampler=self.args.train_sampler,
                                                  num_iters=self.num_train_iter,
                                                  num_epochs=self.epochs,
                                                  num_workers=self.args.num_workers,
                                                  distributed=self.distributed,
                                                  sample_num=sample_num)

        loader_dict['train_ulb'] = get_data_loader(self.args,
                                                   self.dataset_dict['train_ulb'],
                                                   self.args.batch_size * self.args.uratio,
                                                   data_sampler=self.args.train_sampler,
                                                   num_iters=self.num_train_iter,
                                                   num_epochs=self.epochs,
                                                   num_workers=2 * self.args.num_workers,
                                                   distributed=self.distributed,
                                                   sample_num=sample_num)

        loader_dict['eval'] = get_data_loader(self.args,
                                              self.dataset_dict['eval'],
                                              self.args.eval_batch_size,
                                              # make sure data_sampler is None for evaluation
                                              data_sampler=None,
                                              num_workers=self.args.num_workers,
                                              drop_last=False,
                                              tp_record=True)
        
        if self.args.use_tp:
            loader_dict['train_ulb_tp'] = get_data_loader(
                                            self.args,
                                            self.dataset_dict['train_ulb'],
                                            self.args.eval_batch_size,
                                            # make sure data_sampler is None for evaluation
                                            data_sampler=None,
                                            num_workers=self.args.num_workers,
                                            drop_last=False,
                                            tp_record=True,
                                            )

        if self.dataset_dict['test'] is not None:
            loader_dict['test'] =  get_data_loader(self.args,
                                                   self.dataset_dict['test'],
                                                   self.args.eval_batch_size,
                                                   # make sure data_sampler is None for evaluation
                                                   data_sampler=None,
                                                   num_workers=self.args.num_workers,
                                                   drop_last=False)
        self.print_fn(f'[!] data loader keys: {loader_dict.keys()}')
        return loader_dict

    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    self.num_train_iter,
                                                    num_warmup_steps=self.args.num_warmup_iter)
        return optimizer, scheduler

    def set_model(self):
        """
        initialize model
        """
        if not self.args.net_from_name:
            model = self.net_builder(num_classes=self.num_classes, pretrained=self.args.use_pretrain, pretrained_path=self.args.pretrain_path)
        else: # resnet18
            model = self.net_builder(pretrained=self.args.use_pretrain, num_classes=self.num_classes)
        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_hooks(self):
        """
        register necessary training hooks
        """
        # parameter update hook is called inside each train_step
        self.register_hook(ParamUpdateHook(), None, "HIGHEST")
        self.register_hook(EMAHook(), None, "HIGH")
        self.register_hook(EvaluationHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "NORMAL")
        self.register_hook(TimerHook(), None, "LOW")
        self.register_hook(LoggingHook(), None, "LOWEST")
        if self.args.use_wandb:
            self.register_hook(WANDBHook(), None, "LOWEST")
        if self.args.use_aim:
            self.register_hook(AimHook(), None, "LOWEST")

        self.register_hook(CAL_TP(), "TP_record")
        if self.args.use_tp:
            self.register_hook(RecordHook(), "RecordHook")

    def process_batch(self, input_args=None, **kwargs):
        """
        process batch data, send data to cuda
        NOTE **kwargs should have the same arguments to train_step function as keys to work properly
        """
        if input_args is None:
            input_args = signature(self.train_step).parameters
            input_args = list(input_args.keys())

        input_dict = {}

        for arg, var in kwargs.items():
            if not arg in input_args:
                continue
            
            if var is None:
                continue
            
            # send var to cuda
            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.gpu)
            input_dict[arg] = var
        return input_dict
    

    def process_out_dict(self, out_dict=None, **kwargs):
        """
        process the out_dict as return of train_step
        """
        if out_dict is None:
            out_dict = {}

        for arg, var in kwargs.items():
            out_dict[arg] = var
        
        # process res_dict, add output from res_dict to out_dict if necessary
        return out_dict


    def process_log_dict(self, log_dict=None, prefix='train', **kwargs):
        """
        process the tb_dict as return of train_step
        """
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f'{prefix}/' + arg] = var
        return log_dict

    def compute_prob(self, logits):
        return torch.softmax(logits, dim=-1)

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        """
        train_step specific to each algorithm
        """
        # implement train step for each algorithm
        # compute loss
        # update model 
        # record log_dict
        # return log_dict
        raise NotImplementedError


    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        if self.args.tp_gap > 1000:
            self.call_hook("calculate_tp", "TP_record")

        if self.algorithm == 'modis':
            self.call_hook("modis_get_pseudo", "TP_record")
        
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            
            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break
                
                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1
            
            # keep origin_alg_mask consistency in distributed setting
            if self.distributed and epoch == self.epochs-1:
                gathered_masks = [torch.zeros_like(self.origin_alg_mask) for _ in range(self.world_size)]
                gathered_y = [torch.zeros_like(self.origin_alg_y) for _ in range(self.world_size)]
                torch.distributed.all_gather(gathered_masks, self.origin_alg_mask)
                torch.distributed.all_gather(gathered_y, self.origin_alg_y)

                # Combine the results, ensuring consistency
                for mask, y in zip(gathered_masks, gathered_y):
                    valid_positions = (mask == 1.0)
                    self.origin_alg_mask[valid_positions] = True
                    self.origin_alg_y[valid_positions] = y[valid_positions]
                
                origin_pseudo_num = (self.origin_alg_mask>0).sum().item()
                self.print_fn(f"total origin pseudo num: {origin_pseudo_num}")
            
            self.call_hook("after_train_epoch")
        
        self.call_hook("after_run")

    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        # truncate the dummy elements added by SequentialDistributedSampler
        return concat[:num_total_examples]

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):  # 推理全部unlabel数据
        """
        evaluation function
        """
        self.model.eval()
        self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_probs = []
        y_logits = []
        with torch.no_grad():
            for i, data in enumerate(eval_loader):
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                logits = self.model(x)[out_key]
                loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)

                if not self.args.distributed:
                    y_true.extend(y.cpu().tolist())
                    y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                    y_logits.append(logits.cpu().numpy())
                    y_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
                    total_loss += loss.item() * num_batch
                else:
                    y_true.append(y)
                    y_pred.append(torch.max(logits, dim=-1)[1])
                    y_logits.append(logits)
                    y_probs.append(torch.softmax(logits, dim=-1))

                    total_loss += loss * num_batch

        if self.args.distributed:
            # Gather predictions and labels from all processes
            y_true = self.distributed_concat(torch.cat(y_true), len(eval_loader.dataset)).cpu().tolist()
            y_pred = self.distributed_concat(torch.cat(y_pred), len(eval_loader.dataset)).cpu().tolist()
            y_logits = self.distributed_concat(torch.cat(y_logits), len(eval_loader.dataset)).cpu().numpy()
            y_probs = self.distributed_concat(torch.cat(y_probs), len(eval_loader.dataset)).cpu().tolist()
            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_probs, k=5)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()

        eval_dict = {eval_dest+'/loss': total_loss / len(eval_loader.dataset) , eval_dest+'/top-1-acc': top1, eval_dest+'/top-5-acc': top5, 
                     eval_dest+'/balanced_acc': balanced_top1, eval_dest+'/precision': precision, eval_dest+'/recall': recall, eval_dest+'/F1': F1}
        if return_logits:
            eval_dict[eval_dest+'/logits'] = y_logits
        return eval_dict

    def evaluate_all(self, eval_dest='train_ulb_tp', out_key='logits'):  # 推理全部unlabel数据
        """
        evaluation function
        """
        self.model.eval()
        self.ema.apply_shadow()
        
        from tqdm import tqdm
        eval_loader = self.loader_dict[eval_dest]

        logits_all, idx_all = [], []
        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_loader)):
                x = data['x_ulb_w']
                y = data['y_ulb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                
                idx_ulb = data['idx_ulb'].cuda(self.gpu)
                if self.it == 0:  # 记录unlabel数据真实标签
                    self.real_label[idx_ulb] = y.cuda(self.gpu)

                logits = self.model(x)[out_key]
                # 在每个epoch都更新预测标签
                self.pre_label[idx_ulb] = logits.argmax(dim=1)
                
                logits_all.append(logits)
                idx_all.append(data['idx_ulb'].cuda(self.gpu))

            if self.args.distributed:
                logits_all = self.distributed_concat(torch.cat(logits_all), len(eval_loader.dataset))
                # idx_all = self.distributed_concat(torch.cat(idx_all), len(eval_loader.dataset)) # just for debug
                if self.it == 0:
                    torch.distributed.all_reduce(self.real_label, op=torch.distributed.ReduceOp.SUM)
            else:
                logits_all = torch.cat(logits_all)
        
        self.ema.restore()
        self.model.train()
        return logits_all

    def get_save_dict(self):
        """
        make easier for saving model when need save additional arguments
        """
        # base arguments for all models
        save_dict = {
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_scaler': self.loss_scaler.state_dict(),
            'it': self.it + 1,
            'epoch': self.epoch + 1,
            'best_it': self.best_it,
            'best_eval_acc': self.best_eval_acc,
            'origin_alg_mask': self.origin_alg_mask,
            'origin_alg_y': self.origin_alg_y
        }

        if self.args.use_tp:
            tp_save_dict = {}
            tp_save_dict['sum_softmax'] = self.sum_softmax
            tp_save_dict['delta_softmax'] = self.delta_softmax
            tp_save_dict['pre_label'] = self.pre_label
            tp_save_dict['last_pre'] = self.last_pre
            tp_save_dict['real_label'] = self.real_label
            tp_save_dict['training_dynamic'] = self.training_dynamic
            tp_save_dict['tp_pseudo_dict'] = self.tp_pseudo_dict
            tp_save_dict['origin_alg_mask'] = self.origin_alg_mask
            tp_save_dict['origin_alg_y'] = self.origin_alg_y
        else:
            tp_save_dict = None

        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        return save_dict, tp_save_dict
    

    def save_model(self, save_name, save_path):
        """
        save model and specified parameters for resume
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_filename = os.path.join(save_path, save_name)
        save_dict, tp_save_dict = self.get_save_dict()
        torch.save(save_dict, save_filename)
        if tp_save_dict is not None:
            tp_path = os.path.join(save_path, 'tp_'+save_name)
            torch.save(tp_save_dict, tp_path)
            self.print_fn(f"tp info saved: {tp_path}")
        self.print_fn(f"model saved: {save_filename}")


    def load_model(self, load_path, tp_info_path=None):
        """
        load model and specified parameters for resume
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        self.it = checkpoint['it']
        self.start_epoch = checkpoint['epoch']
        self.epoch = self.start_epoch
        self.best_it = checkpoint['best_it']
        self.best_eval_acc = checkpoint['best_eval_acc']

        if tp_info_path is not None:
            tp_info = torch.load(tp_info_path, map_location='cpu')
            self.sum_softmax = tp_info['sum_softmax'].cuda(self.gpu)
            self.delta_softmax = tp_info['delta_softmax'].cuda(self.gpu)
            self.pre_label = tp_info['pre_label'].cuda(self.gpu)
            self.last_pre = tp_info['last_pre'].cuda(self.gpu)
            self.real_label = tp_info['real_label'].cuda(self.gpu)
            self.training_dynamic = tp_info['training_dynamic']
            self.tp_pseudo_dict = tp_info['tp_pseudo_dict']
            self.best_eval_acc = 0
        
        self.origin_alg_mask = checkpoint['origin_alg_mask'].cuda(self.gpu)
        self.origin_alg_y = checkpoint['origin_alg_y'].cuda(self.gpu)

        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # if self.scheduler is not None and 'scheduler' in checkpoint:
        #     self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.print_fn('Model loaded')
        return checkpoint

    def single_load_tp(self, tp_info_path):
        tp_info = torch.load(tp_info_path, map_location='cpu')
        self.sum_softmax = tp_info['sum_softmax'].cuda(self.gpu)
        self.delta_softmax = tp_info['delta_softmax'].cuda(self.gpu)
        self.pre_label = tp_info['pre_label'].cuda(self.gpu)
        self.last_pre = tp_info['last_pre'].cuda(self.gpu)
        self.real_label = tp_info['real_label'].cuda(self.gpu)
        self.training_dynamic = tp_info['training_dynamic']
        self.tp_pseudo_dict = tp_info['tp_pseudo_dict']
        self.origin_alg_mask = tp_info['origin_alg_mask'].cuda(self.gpu)
        self.origin_alg_y = tp_info['origin_alg_y'].cuda(self.gpu)
        self.print_fn('tp info loaded')

    def check_prefix_state_dict(self, state_dict):
        """
        remove prefix state dict in ema model
        """
        new_state_dict = dict()
        for key, item in state_dict.items():
            if key.startswith('module'):
                new_key = '.'.join(key.split('.')[1:])
            else:
                new_key = key
            new_state_dict[new_key] = item
        return new_state_dict

    def register_hook(self, hook, name=None, priority='NORMAL'):
        """
        Ref: https://github.com/open-mmlab/mmcv/blob/a08517790d26f8761910cac47ce8098faac7b627/mmcv/runner/base_runner.py#L263
        Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            hook_name (:str, default to None): Name of the hook to be registered. Default is the hook class name.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        hook.name = name if name is not None else type(hook).__name__

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        
        if not inserted:
            self._hooks.insert(0, hook)

        # call set hooks
        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook
        


    def call_hook(self, fn_name, hook_name=None, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            hook_name (str): The specific hook name to be called, such as
                "param_update" or "dist_align", uesed to call single hook in train_step.
        """
        
        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)
        
        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)

    def registered_hook(self, hook_name):
        """
        Check if a hook is registered
        """
        return hook_name in self.hooks_dict


    @staticmethod
    def get_argument():
        """
        Get specificed arguments into argparse for each algorithm
        """
        return {}



class ImbAlgorithmBase(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        
        # imbalanced arguments
        self.lb_imb_ratio = self.args.lb_imb_ratio
        self.ulb_imb_ratio = self.args.ulb_imb_ratio
        self.imb_algorithm = self.args.imb_algorithm
    
    def imb_init(self, *args, **kwargs):
        """
        intiialize imbalanced algorithm parameters
        """
        pass 

    def set_optimizer(self):
        if 'vit' in self.args.net and self.args.dataset in ['cifar100', 'food101', 'semi_aves', 'semi_aves_out']:
            return super().set_optimizer() 
        elif self.args.dataset in ['imagenet', 'imagenet127']:
            return super().set_optimizer() 
        else:
            self.print_fn("Create optimizer and scheduler")
            optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay, bn_wd_skip=False)
            scheduler = None
            return optimizer, scheduler
