# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, PseudoRecorder
from semilearn.algorithms.utils import SSL_Argument


@ALGORITHMS.register('modis')
class MoDis(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.init(p_cutoff=args.p_cutoff, unsup_warm_up=args.unsup_warm_up)
        self.modis_weight = nn.Parameter(torch.ones(1, device=self.gpu) * 0.05)

    def init(self, p_cutoff, unsup_warm_up=0.4):
        self.p_cutoff = p_cutoff
        self.unsup_warm_up = unsup_warm_up 

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        self.register_hook(PseudoRecorder(), "PseudoRecorderHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w):
        modis_pseudo_dict = self.modis_pseudo_dict
        # print('modis number:', modis_pseudo_dict['modis_idx_num'])
        modis_idx = modis_pseudo_dict['modis_idx']
        modis_now_batch = list(set(modis_idx) & set(idx_ulb))

        # inference and calculate sup/unsup losses
        with self.amp_cm(dtype=torch.bfloat16):
            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']

            outs_x_ulb = self.model(x_ulb_w)
            logits_x_ulb = outs_x_ulb['logits']
            feats_x_ulb = outs_x_ulb['feat']
            
            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                        logits=logits_x_ulb,
                                        use_hard_label=True)

            # record pseudo-labeling statistics and return by out_dict
            pseudo_idx, util_pseudo = self.call_hook("get_pseduo", "PseudoRecorderHook",
                        idx_ulb=idx_ulb,
                        mask=mask,
                        pseudo_label=pseudo_label)
            self.origin_alg_mask[pseudo_idx] = 1.0  
            self.origin_alg_y[pseudo_idx] = util_pseudo  

            if self.args.use_tp:
                self.pre_label[pseudo_idx] = util_pseudo

            # NOTE:set a large tp_gap when contine train two phase
            if self.args.use_tp:
                self.call_hook("record_train_dynamic", "RecordHook", n=self.args.tp_gap) 

            if self.tp_pseudo_dict is not None:
                neg_idx, neg_mask, tp_num, negetive_num, tp_pseudo_acc, tp_loss, loss_nl = self.call_hook(
                    "calculate_loss", 
                    "TP_record",
                    idx_ulb, 
                    mask,
                    logits_x_ulb
                )      
            else:
                neg_idx = None
                neg_mask = None
                tp_num = 0
                negetive_num = 0
                tp_pseudo_acc = 0
                tp_loss = 0 
                loss_nl = 0      

            unsup_loss = self.consistency_loss(logits_x_ulb.float(),
                                               pseudo_label,
                                               'ce',
                                               mask=mask,
                                               )
            
            if len(modis_now_batch) > 0:
                modis_loss = F.cross_entropy(logits_x_ulb[modis_now_batch], pseudo_label[modis_now_batch])
            else:
                modis_loss = 0

            unsup_warmup = np.clip(self.it / (self.unsup_warm_up * self.num_train_iter),  a_min=0.0, a_max=1.0)
            total_loss = sup_loss + self.lambda_u * (unsup_loss+tp_loss+loss_nl) * unsup_warmup + modis_loss*self.modis_weight

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict, pseudo_idx=pseudo_idx, 
                                         util_pseudo=None, logits_x_ulb_w=logits_x_ulb)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item(),
                                         total_origin_pse=(self.origin_alg_mask>0).sum().item(),
                                         tp_num=tp_num,
                                         negetive_num=negetive_num,
                                         tp_pseudo_acc=tp_pseudo_acc)
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--unsup_warm_up', float, 0.4, 'warm up ratio for unsupervised loss'),
            SSL_Argument("--modis_T", type=int, default=0.1),
            SSL_Argument("--modis_thres", type=float, default=4)
            # SSL_Argument('--use_flex', str2bool, False),
        ]