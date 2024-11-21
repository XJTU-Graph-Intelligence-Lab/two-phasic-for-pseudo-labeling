# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import logging
import json
import os
import random
import warnings
import wandb
import traceback

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
from semilearn.algorithms import get_algorithm, name2alg
from semilearn.core.utils import (
    TBLog,
    count_parameters,
    get_logger,
    get_net_builder,
    get_port,
    over_write_args_from_file,
    send_model_cuda,
)
from semilearn.imb_algorithms import get_imb_algorithm, name2imbalg

import optuna
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def get_config():
    from semilearn.algorithms.utils import str2bool

    parser = argparse.ArgumentParser(description="Semi-Supervised Learning (USB)")

    """
    Saving & loading of the model.
    """
    parser.add_argument("--save_dir", type=str, default="/mnt/bn/liyullm2/liyussl/saved_models")
    parser.add_argument("-sn", "--save_name", type=str, default="fixmatch")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--tp_info_path", type=str, default=None)
    parser.add_argument("--resume_all", type=bool, default=True)
    parser.add_argument("-o", "--overwrite", action="store_true", default=True)
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        help="Use tensorboard to plot and save curves",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use wandb to plot and save curves"
    )
    parser.add_argument(
        "--use_aim", action="store_true", help="Use aim to plot and save curves"
    )

    """
    two-phase
    """
    parser.add_argument("--use_tp", type=bool, default=False)
    parser.add_argument("--tp_epoch", type=int, default=1500)
    parser.add_argument("--tp_gap", type=int, default=25)
    parser.add_argument("--e_epoch", type=int, default=200)
    parser.add_argument('--new_sample_strategy', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--eta1', type=float, default=0.5)
    parser.add_argument('--eta2', type=float, default=0.5)
    parser.add_argument('--eta3', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.1)
    parser.add_argument('--b2', type=float, default=0.1)
    parser.add_argument('--neg', type=float, default=0.004)
    parser.add_argument('--tp_thres', type=float, default=2)
    parser.add_argument('--is_mini_data', type=bool, default=False)
    """
    Training Configuration of FixMatch
    """
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument(
        "--num_train_iter",
        type=int,
        default=20,
        help="total number of training iterations",
    )
    parser.add_argument(
        "--num_warmup_iter", type=int, default=0, help="cosine linear warmup iterations"
    )
    parser.add_argument(
        "--num_eval_iter", type=int, default=1, help="evaluation frequency"
    )
    parser.add_argument("--num_log_iter", type=int, default=5, help="logging frequency")
    parser.add_argument("-nl", "--num_labels", type=int, default=400)
    parser.add_argument("-bsz", "--batch_size", type=int, default=8)
    parser.add_argument(
        "--uratio",
        type=int,
        default=1,
        help="the ratio of unlabeled data to labeled data in each mini-batch",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="batch size of evaluation data loader (it does not affect the accuracy)",
    )
    parser.add_argument(
        "--ema_m", type=float, default=0.999, help="ema momentum for eval_model"
    )
    parser.add_argument("--ulb_loss_ratio", type=float, default=1.0)

    """
    Optimizer configurations
    """
    parser.add_argument("--optim", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=1.0,
        help="layer-wise learning rate decay, default to 1.0 which means no layer "
        "decay",
    )

    """
    Backbone Net Configurations
    """
    parser.add_argument("--net", type=str, default="wrn_28_2")
    parser.add_argument("--net_from_name", type=str2bool, default=False)
    parser.add_argument("--use_pretrain", default=False, type=str2bool)
    parser.add_argument("--pretrain_path", default="", type=str)

    """
    Algorithms Configurations
    """

    ## core algorithm setting
    parser.add_argument(
        "-alg", "--algorithm", type=str, default="fixmatch", help="ssl algorithm"
    )
    parser.add_argument(
        "--use_cat", type=str2bool, default=True, help="use cat operation in algorithms"
    )
    parser.add_argument(
        "--amp",
        type=str2bool,
        default=False,
        help="use mixed precision training or not",
    )
    parser.add_argument("--clip_grad", type=float, default=0)

    ## imbalance algorithm setting
    parser.add_argument(
        "-imb_alg",
        "--imb_algorithm",
        type=str,
        default=None,
        help="imbalance ssl algorithm",
    )

    """
    Data Configurations
    """

    ## standard setting configurations
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("-ds", "--dataset", type=str, default="cifar10")
    parser.add_argument("-nc", "--num_classes", type=int, default=10)
    parser.add_argument("--train_sampler", type=str, default="RandomSampler")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--include_lb_to_ulb",
        type=str2bool,
        default="True",
        help="flag of including labeled data into unlabeled data, default to True",
    )

    ## imbalanced setting arguments
    parser.add_argument(
        "--lb_imb_ratio",
        type=int,
        default=1,
        help="imbalance ratio of labeled data, default to 1",
    )
    parser.add_argument(
        "--ulb_imb_ratio",
        type=int,
        default=1,
        help="imbalance ratio of unlabeled data, default to 1",
    )
    parser.add_argument(
        "--ulb_num_labels",
        type=int,
        default=None,
        help="number of labels for unlabeled data, used for determining the maximum "
        "number of labels in imbalanced setting",
    )

    ## cv dataset arguments
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--crop_ratio", type=float, default=0.875)

    ## nlp dataset arguments
    parser.add_argument("--max_length", type=int, default=512)

    ## speech dataset algorithms
    parser.add_argument("--max_length_seconds", type=float, default=4.0)
    parser.add_argument("--sample_rate", type=int, default=16000)

    """
    multi-GPUs & Distributed Training
    """

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)  # noqa: E501
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="**node rank** for distributed training"
    )
    parser.add_argument(
        "-du",
        "--dist-url",
        default="tcp://127.0.0.1:11111",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--seed", default=1, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--multiprocessing-distributed",
        type=str2bool,
        default=False,
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    # config file
    parser.add_argument("--c", type=str, default="config/two_phase/fixmatch/cifar_tp400.yaml")

    # add algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    for argument in name2alg[args.algorithm].get_argument():
        parser.add_argument(
            argument.name,
            type=argument.type,
            default=argument.default,
            help=argument.help,
        )

    # add imbalanced algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    if args.imb_algorithm is not None:
        for argument in name2imbalg[args.imb_algorithm].get_argument():
            parser.add_argument(
                argument.name,
                type=argument.type,
                default=argument.default,
                help=argument.help,
            )
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    return args


def main(args, trial):
    """
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    """
    if not args.new_sample_strategy:
        assert (
            args.num_train_iter % args.epoch == 0
        ), f"# total training iter. {args.num_train_iter} is not divisible by # epochs {args.epoch}"  # noqa: E501

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and args.resume is False:
        import shutil

        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception("already existing model: {}".format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception("Resume of training requires --load_path in the args")
        if (
            os.path.abspath(save_path) == os.path.abspath(args.load_path)
            and not args.overwrite
        ):
            raise Exception(
                "Saving & Loading paths are same. \
                            If you want over-write, give --overwrite in the argument."
            )

    if args.seed is not None:
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu == "None":
        args.gpu = None
    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

    best_acc = float(json.load(open('res.json', 'r')))
    return best_acc


def main_worker(gpu, ngpus_per_node, args):
    """
    main_worker is conducted on each GPU.
    """

    global best_acc1
    args.gpu = gpu

    # random seed has to be set for the synchronization of labeled data sampling in each
    # process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # set distributed group:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, "tensorboard", use_tensorboard=args.use_tensorboard)
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.info(f"Use GPU: {args.gpu} for training")

    _net_builder = get_net_builder(args.net, args.net_from_name)
    # optimizer, scheduler, datasets, dataloaders with be set in algorithms
    if args.imb_algorithm is not None:
        model = get_imb_algorithm(args, _net_builder, tb_log, logger)
    else:
        model = get_algorithm(args, _net_builder, tb_log, logger)
    logger.info(f"Number of Trainable Params: {count_parameters(model.model)}")

    # SET Devices for (Distributed) DataParallel
    model.model = send_model_cuda(args, model.model)
    model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)
    logger.info(f"Arguments: {model.args}")

    # If args.resume, load checkpoints from args.load_path
    if args.resume and os.path.exists(args.load_path):
        if args.resume_all:
            try:
                model.load_model(args.load_path, args.tp_info_path)
            except:
                logger.info("Fail to resume load path {}".format(args.load_path))
                args.resume = False
        else:
            try:
                model.single_load_tp(args.tp_info_path)
            except:
                logger.info("Fail to resume load path {}".format(args.tp_info_path))
                args.resume = False
    else:
        logger.info("Resume load path {} does not exist".format(args.load_path))

    if hasattr(model, "warmup"):
        logger.info(("Warmup stage"))
        model.warmup()

    # START TRAINING of FixMatch
    logger.info("Model training")
    model.train()

    # print validation (and test results)
    for key, item in model.results_dict.items():
        logger.info(f"Model result - {key} : {item}")

    if hasattr(model, "finetune"):
        logger.info("Finetune stage")
        model.finetune()

    dist.destroy_process_group()
    logging.warning(f"GPU {args.rank} training is FINISHED")

    if gpu == 0:
        with open('res.json', 'w') as f:
            json.dump(model.results_dict['eval/best_acc'], f)
                   

def set_search_space(trial):
    args.lr = trial.suggest_float('lr', 0.0001, 0.001, log=True)
    args.layer_decay = trial.suggest_float('weight_decay', 0.1, 0.9, step=0.1)
    args.alpha = trial.suggest_float('alpha', 0, 0.5)
    args.beta = trial.suggest_float('beta', 0.5, 1)
    args.eta1 = trial.suggest_float('eta1', 0, 1)
    args.eta2 = trial.suggest_float('eta2', 0, 1)
    args.eta3 = trial.suggest_float('eta3', 0, 1)
    args.b1 = trial.suggest_float('b1', 0, 0.1)
    args.b2 = trial.suggest_float('b2', 0.1, 0.2)
    args.tp_thres = trial.suggest_float('tp_thres', 1.5, 40)
    
    import time; time.sleep(5)
    args.world_size = 1
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    best_acc = main(args, trial)
    return best_acc


def runner(study_name, storage_name, n_trials):
    try:
        study = optuna.create_study(study_name=study_name, direction='maximize', storage=storage_name, load_if_exists=True)
        study.optimize(set_search_space, n_trials=n_trials)
        wandb.alert(
            title="Optuna Optimization Alert",
            text=f"Optimization {study_name} finsh"
        )
    except:
        wandb.alert(
            title="Optuna Optimization Alert",
            text=f"Optimization failed Error: {traceback.format_exc()}"
        )
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        # 计算剩余的 trials 数量
        completed_trials = len(study.trials)
        remaining_trials = max(0, n_trials - completed_trials)
            
        if remaining_trials > 0:
            study.optimize(set_search_space, n_trials=remaining_trials)
            wandb.alert(
                title="Optuna Optimization Alert",
                text=f"Optimization {study_name} finsh"
            )

    return study


if __name__ == "__main__":
    args = get_config()
    base_save_name = args.save_name
    port = get_port()
    args.dist_url = "tcp://127.0.0.1:" + str(port)

    n_trials = 30
    save_path = os.path.join(args.save_dir, args.save_name)
    study_name = f"{args.algorithm}-{args.save_name}" 
    storage_name = "sqlite:///{}.db".format(os.path.join(args.save_dir, study_name))
    
    os.environ['MASTER_PORT'] = str(port+1)
    
    wandb.init(project="optuna", name=study_name)
    study = runner(study_name, storage_name, n_trials)

    beat_res = study.best_params
    beat_res['value'] = study.best_value
    with open(os.path.join(save_path, f'optuna+tp_weight_loss_thres_{args.algorithm}_{args.dataset}_{args.num_labels}_best_results.json'), 'w') as outfile:
        json.dump(beat_res, outfile, indent=4)
    
    all_res = []
    for trial in study.trials:
        res = {}
        res['number'] = trial.number    
        res['params'] = trial.params
        res['value'] = trial.value
        all_res.append(res)
    with open(os.path.join(save_path, f'optuna+tp_weight_loss_thres_{args.algorithm}_{args.dataset}_{args.num_labels}_all_results.json'), 'w') as outfile:
        json.dump(all_res, outfile, indent=4)

    # Visualize the optimization history.
    fig1 = plot_optimization_history(study)
    fig1.write_image(os.path.join(save_path, 'history.jpg'))

    # Visualize individual hyperparameters.
    fig2 = plot_slice(study)
    fig2.write_image(os.path.join(save_path,'slice.jpg'))

    # Select parameter
    # Visualize parameter importances.
    fig3 = plot_param_importances(study)
    fig3.write_image(os.path.join(save_path,'importances.jpg'))

    # Select parameters to visualize.
    # fig4 = plot_parallel_coordinate(study, params=["alpha", "beta"])
    # fig4.write_image(os.path.join(save_path,'parallel.jpg'))