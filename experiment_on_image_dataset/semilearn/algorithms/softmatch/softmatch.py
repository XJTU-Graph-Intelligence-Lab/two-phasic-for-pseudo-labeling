# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from .utils import SoftMatchWeightingHook
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook, PseudoRecorder
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('softmatch')
class SoftMatch(AlgorithmBase):
    """
        SoftMatch algorithm (https://openreview.net/forum?id=ymt1zQXBDiF&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2023%2FConference%2FAuthors%23your-submissions)).

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
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ema_p (`float`):
                exponential moving average of probability update
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, dist_align=args.dist_align, dist_uniform=args.dist_uniform, ema_p=args.ema_p, n_sigma=args.n_sigma, per_class=args.per_class)
    
    def init(self, T, hard_label=True, dist_align=True, dist_uniform=True, ema_p=0.999, n_sigma=2, per_class=False):
        self.T = T
        self.use_hard_label = hard_label
        self.dist_align = dist_align
        self.dist_uniform = dist_uniform
        self.ema_p = ema_p
        self.n_sigma = n_sigma
        self.per_class = per_class

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='uniform' if self.args.dist_uniform else 'model'), 
            "DistAlignHook")
        self.register_hook(SoftMatchWeightingHook(num_classes=self.num_classes, n_sigma=self.args.n_sigma, momentum=self.args.ema_p, per_class=self.args.per_class), "MaskingHook")
        self.register_hook(PseudoRecorder(), "PseudoRecorderHook")
        super().set_hooks()    

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm(dtype=torch.bfloat16):
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

            probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

            # uniform distribution alignment 
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)

            if self.args.use_tp:
                self.call_hook("record_train_dynamic", "RecordHook", n=self.args.tp_gap) 

            # NOTE:set tp_epoch=0 when contine train two phase
            # if self.args.tp_gap<1000 and (self.epoch <= self.args.tp_epoch-1 or self.args.use_tp is False):
            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                        # make sure this is logits, not dist aligned probs
                                        # uniform alignment in softmatch do not use aligned probs for generating pesudo labels
                                        logits=logits_x_ulb_w,
                                        use_hard_label=self.use_hard_label,
                                        T=self.T)
            # record pseudo-labeling statistics and return by out_dict
            pseudo_idx, util_pseudo = self.call_hook("get_pseduo", "PseudoRecorderHook",
                        idx_ulb=idx_ulb,
                        mask=mask,
                        pseudo_label=pseudo_label)
            self.origin_alg_mask[pseudo_idx] = 1.0  
            self.origin_alg_y[pseudo_idx] = util_pseudo
            
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
                logits_x_ulb_w
            )  
            # calculate loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)

            total_loss = sup_loss + self.lambda_u * (unsup_loss+tp_loss+loss_nl)

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict, pseudo_idx=pseudo_idx, 
                                         util_pseudo=util_pseudo, logits_x_ulb_w=logits_x_ulb_w)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item(),
                                         total_origin_pse=(self.origin_alg_mask>0).sum().item(),
                                         tp_num=tp_num,
                                         negetive_num=negetive_num,
                                         tp_pseudo_acc=tp_pseudo_acc)
        return out_dict, log_dict

    # TODO: change these
    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict[0]['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict[0]['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        save_dict[0]['prob_max_mu_t'] = self.hooks_dict['MaskingHook'].prob_max_mu_t.cpu()
        save_dict[0]['prob_max_var_t'] = self.hooks_dict['MaskingHook'].prob_max_var_t.cpu()
        return save_dict


    def load_model(self, load_path, tp_info_path=None):
        checkpoint = super().load_model(load_path, tp_info_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_mu_t = checkpoint['prob_max_mu_t'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_var_t = checkpoint['prob_max_var_t'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--dist_uniform', str2bool, True),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--n_sigma', int, 2),
            SSL_Argument('--per_class', str2bool, False),
        ]
