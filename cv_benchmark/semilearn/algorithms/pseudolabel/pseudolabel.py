# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, PseudoRecorder
from semilearn.algorithms.utils import SSL_Argument


@ALGORITHMS.register('pseudolabel')
class PseudoLabel(AlgorithmBase):
    """
        Pseudo Label algorithm (https://arxiv.org/abs/1908.02983).

        Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - p_cutoff(`float`):
            Confidence threshold for generating pseudo-labels
        - unsup_warm_up (`float`, *optional*, defaults to 0.4):
            Ramp up for weights for unsupervised loss
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.init(p_cutoff=args.p_cutoff, unsup_warm_up=args.unsup_warm_up)

    def init(self, p_cutoff, unsup_warm_up=0.4):
        self.p_cutoff = p_cutoff
        self.unsup_warm_up = unsup_warm_up 

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        self.register_hook(PseudoRecorder(), "PseudoRecorderHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w):
        # inference and calculate sup/unsup losses
        with self.amp_cm(dtype=torch.bfloat16):

            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']

            # calculate BN only for the first batch
            self.bn_controller.freeze_bn(self.model)
            outs_x_ulb = self.model(x_ulb_w)
            logits_x_ulb = outs_x_ulb['logits']
            feats_x_ulb = outs_x_ulb['feat']
            self.bn_controller.unfreeze_bn(self.model)

            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # NOTE:set a large tp_gap when contine train two phase
            if self.args.use_tp:
                self.call_hook("record_train_dynamic", "RecordHook", n=self.args.tp_gap, save_now=True) 

            # NOTE:set tp_epoch=0 when contine train two phase
            # if self.args.tp_gap<1000 and (self.epoch <= self.args.tp_epoch-1 or self.args.use_tp is False):
            # compute mask
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
            
            if self.epoch > self.args.tp_epoch:
                # get mask from self.origin_alg_mask
                # mask, pseudo_label = self.call_hook("get_mask_from_origin", "TP_record", idx_ulb) 
                pseudo_idx, util_pseudo = None, None
                if self.it % self.args.tp_gap == 0:
                    # update tp info
                    self.call_hook("calculate_tp", "TP_record")

            neg_idx, neg_mask, tp_num, negetive_num, tp_pseudo_acc, tp_loss, loss_nl = self.call_hook(
                "calculate_loss", 
                "TP_record",
                idx_ulb, 
                mask,
                logits_x_ulb
            )             

            # unsup_loss = self.consistency_loss(logits_x_ulb.float(),
            #                                    pseudo_label,
            #                                    'ce',
            #                                    mask=mask,
            #                                    )
            unsup_loss = torch.tensor(0).cuda(self.gpu)
            unsup_warmup = np.clip(self.it / (self.unsup_warm_up * self.num_train_iter),  a_min=0.0, a_max=1.0)
            total_loss = sup_loss + self.lambda_u * (unsup_loss+tp_loss+loss_nl) * unsup_warmup

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict, pseudo_idx=pseudo_idx, 
                                         util_pseudo=util_pseudo, logits_x_ulb_w=logits_x_ulb)
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
            # SSL_Argument('--use_flex', str2bool, False),
        ]