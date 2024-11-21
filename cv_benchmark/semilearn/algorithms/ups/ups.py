import torch
import numpy as np
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, PseudoRecorder
from semilearn.algorithms.utils import SSL_Argument


@ALGORITHMS.register('ups')
class UPS(AlgorithmBase):
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

    def caculate_neg_loss(self, logits_x_ulb, other_outs):
        out = torch.cat((logits_x_ulb.unsqueeze(0), other_outs.view(5, -1, self.args.num_classes)))

        out_prob_nl = torch.softmax(out/self.args.T, dim=-1)
        out_std_nl = torch.std(out_prob_nl, dim=0)
        out_prob_nl = torch.mean(out_prob_nl, dim=0)
        nl_mask = ((out_std_nl < self.args.kappa_n) * (out_prob_nl < self.args.tau_n))

        pred_nl = torch.softmax(logits_x_ulb, dim=1)
        pred_nl = 1 - pred_nl
        pred_nl = torch.clamp(pred_nl, 1e-7, 1.0)

        y_nl = torch.ones((logits_x_ulb.shape)).to(device=self.gpu, dtype=logits_x_ulb.dtype)
        loss_nl = torch.mean((-torch.sum((y_nl * torch.log(pred_nl))*nl_mask, dim = -1))/(torch.sum(nl_mask, dim = -1) + 1e-7))
        return loss_nl
    
    def caculate_pos_loss(self, logits_x_ulb, other_outs, idx_ulb):
        out = torch.cat((logits_x_ulb.unsqueeze(0), other_outs.view(5, -1, self.args.num_classes)))

        out_prob = torch.softmax(out, dim=-1)

        out_std = torch.std(out_prob, dim=0)
        out_prob = torch.mean(out_prob, dim=0)
        
        max_value, max_idx = torch.max(out_prob, dim=1)
        max_std = out_std.gather(1, max_idx.view(-1,1))
        selected_idx = (max_value>=self.args.tau_p) * (max_std.squeeze(1) < self.args.kappa_p)

         # record pseudo-labeling statistics and return by out_dict
        pseudo_idx, util_pseudo = self.call_hook("get_pseduo", "PseudoRecorderHook",
                    idx_ulb=idx_ulb,
                    mask=selected_idx,
                    pseudo_label=max_idx)

        self.origin_alg_mask[pseudo_idx] = 1.0  
        self.origin_alg_y[pseudo_idx] = util_pseudo  
        
        pos_loss = self.consistency_loss(logits_x_ulb, max_idx, 'ce',mask=selected_idx.float())
        return pos_loss, selected_idx.float(), pseudo_idx, util_pseudo
    
    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w):
        # inference and calculate sup/unsup losses
        with self.amp_cm(dtype=torch.bfloat16):
            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']

            # calculate BN only for the first batch
            self.bn_controller.freeze_bn(self.model)
            outs_x_ulb = self.model(x_ulb_w)
            
            # add drop input
            other_input = [x_ulb_w for _ in range(5)]
            other_input = F.dropout(torch.cat(other_input, dim=0), p=0.3) 
            other_logits = self.model(other_input)['logits']

            logits_x_ulb = outs_x_ulb['logits']
            feats_x_ulb = outs_x_ulb['feat']
            self.bn_controller.unfreeze_bn(self.model)

            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
        
            # NOTE:set a large tp_gap when contine train two phase
            if self.args.use_tp:
                self.call_hook("record_train_dynamic", "RecordHook", n=self.args.tp_gap) 

            if self.epoch > self.args.tp_epoch:
                # get mask from self.origin_alg_mask
                # mask, pseudo_label = self.call_hook("get_mask_from_origin", "TP_record", idx_ulb) 
                pseudo_idx, util_pseudo = None, None
                if self.it % self.args.tp_gap == 0:
                    # update tp info
                    self.call_hook("calculate_tp", "TP_record")

            unsup_loss, mask, pseudo_idx, util_pseudo = self.caculate_pos_loss(logits_x_ulb, other_logits, idx_ulb)
            if self.args.use_tp:
                self.pre_label[pseudo_idx] = util_pseudo

            neg_idx, neg_mask, tp_num, negetive_num, tp_pseudo_acc, tp_loss, loss_nl = self.call_hook(
                "calculate_loss", 
                "TP_record",
                idx_ulb, 
                mask,
                logits_x_ulb
            )             

            neg_loss = self.caculate_neg_loss(logits_x_ulb, other_logits)

            unsup_warmup = np.clip(self.it / (self.unsup_warm_up * self.num_train_iter),  a_min=0.0, a_max=1.0)
            total_loss = sup_loss + self.lambda_u * (unsup_loss+tp_loss+loss_nl+neg_loss) * unsup_warmup

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
            SSL_Argument('--T', float, 2.0),
            SSL_Argument('--kappa_n', float, 0.005),
            SSL_Argument('--tau_n', float, 0.05),
            SSL_Argument('--kappa_p', float, 0.05),
            SSL_Argument('--tau_p', float, 0.7)
        ]