# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py

import os
from .hook import Hook


class RecordHook(Hook):
    """
    Evaluation Hook for validation during training
    """
    
    def record_train_dynamic(self, algorithm, n, save_now=False):
        if algorithm.args.use_tp and n < 1500 and algorithm.record_id<algorithm.num_record: # when two pahse contine train, set n > 100
            if self.every_n_iters(algorithm, n) or algorithm.it == 0:
                algorithm.print_fn("record tp all...")
                logits_all = algorithm.evaluate_all('train_ulb_tp')
                algorithm.call_hook("record_td", "TP_record", logits_all)
                algorithm.record_id += 1
                algorithm.print_fn("record tp finish...")

                if save_now:
                    save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
                    algorithm.save_model(f'model_{algorithm.it}_{algorithm.record_id}.pth', save_path)