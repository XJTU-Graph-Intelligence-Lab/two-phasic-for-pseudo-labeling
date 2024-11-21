import torch

from semilearn.core.hooks import Hook

class PseudoRecorder(Hook):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def get_pseduo(self, 
                    algorithm,
                    idx_ulb, 
                    mask, 
                    pseudo_label
                    ):
        valid_idx = mask > 0
        return idx_ulb[valid_idx], pseudo_label[valid_idx]

