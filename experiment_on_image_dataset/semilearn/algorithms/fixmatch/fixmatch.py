# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, PseudoRecorder
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('fixmatch')
class FixMatch(AlgorithmBase):

    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
    
    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        self.register_hook(PseudoRecorder(), "PseudoRecorderHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm(dtype=torch.bfloat16): # use bf16
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # NOTE:set a large tp_gap when contine train two phase
            if self.args.use_tp and self.record_id < self.num_record:
                self.call_hook("record_train_dynamic", "RecordHook", n=self.args.tp_gap)
            
            # NOTE:set tp_epoch=0 when contine train two phase
            # if self.args.tp_gap<1000 and (self.epoch <= self.args.tp_epoch-1 or self.args.use_tp is False):
            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                        logits=probs_x_ulb_w,
                                        use_hard_label=self.use_hard_label,
                                        T=self.T,
                                        softmax=False)
            
            # record pseudo-labeling statistics and return by out_dict
            pseudo_idx, util_pseudo = self.call_hook("get_pseduo", "PseudoRecorderHook",
                        idx_ulb=idx_ulb,
                        mask=mask,
                        pseudo_label=pseudo_label)
            self.origin_alg_mask[pseudo_idx] = 1.0
            self.origin_alg_y[pseudo_idx] = util_pseudo
            
            # in stage3 add tp loss
            if self.epoch > self.args.tp_epoch and self.args.use_tp:
                # get mask from self.origin_alg_mask
                # mask, pseudo_label = self.call_hook("get_mask_from_origin", "TP_record", idx_ulb) 
                pseudo_idx, util_pseudo = None, None
                if self.it % self.args.tp_gap == 0: # old two phase continue train strategy
                    # update tp info
                    self.call_hook("calculate_tp", "TP_record")
            
            neg_idx, neg_mask, tp_num, negetive_num, tp_pseudo_acc, tp_loss, loss_nl = self.call_hook(
                "calculate_loss", 
                "TP_record",
                idx_ulb, 
                mask, 
                logits_x_ulb_w
            )

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask
                                               )

            total_loss = sup_loss + self.lambda_u * (unsup_loss+tp_loss+loss_nl)

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict, pseudo_idx=pseudo_idx, 
                                         util_pseudo=util_pseudo, logits_x_ulb_w=logits_x_ulb_w,
                                         )
        
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
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]