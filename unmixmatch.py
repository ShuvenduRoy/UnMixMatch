# import needed library
import os
import logging
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from utils import net_builder, over_write_args_from_file, get_logger, get_optimizer, get_cosine_schedule_with_warmup
from models.unmixmatch.unmixmatch import UnMixMatch
from datasets.dataset_helpter import get_dataset_and_loader


# from datasets.ssl_dataset import SSL_Dataset, ImageDatasetLoader
# from datasets.data_utils import get_data_loader


def main(args):
    """
    main_worker is conducted on each GPU.
    """

    global best_acc1
    save_path = os.path.join(args.save_dir, args.save_name)

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")

    model = UnMixMatch(net_builder(args.net,
                                   args.net_from_name,
                                   {'first_stride': 2 if 'stl' in args.dataset else 1,
                                    'depth': args.depth,
                                    'widen_factor': args.widen_factor,
                                    'leaky_slope': args.leaky_slope,
                                    'bn_momentum': args.bn_momentum,
                                    'dropRate': args.dropout,
                                    'use_embed': False,
                                    'is_remix': True,
                                    'projection_head': args.projection_head,
                                    'dim_in': args.dim_in,
                                    'feat_dim': args.feat_dim, },
                                   ),
                       args.num_classes,
                       args.ema_m,
                       args.T,
                       num_eval_iter=args.num_eval_iter,
                       logger=logger)

    # SET Optimizer & LR Scheduler
    # construct SGD and cosine lr scheduler
    optimizer = get_optimizer(model.model, args.optim, args.lr, args.momentum, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                args.num_train_iter,
                                                num_warmup_steps=args.num_train_iter * 0)
    # set SGD and cosine lr on RemixMatch
    model.set_optimizer(optimizer, scheduler)

    # SET Devices for (Distributed) DataParallel
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model.model.cuda(args.gpu)
        cudnn.benchmark = True

    dset_dict, loader_dict = get_dataset_and_loader(args, ulb_workers_multiplier=2)
    model.set_data_loader(loader_dict)

    # If args.resume, load checkpoints from args.load_path
    if os.path.exists(os.path.join(save_path, 'latest_model.pth')):
        print('Attempting auto-resume!!')
        model.load_model(os.path.join(save_path, 'latest_model.pth'))

    # START TRAINING of RemixMatch
    print(args)
    trainer = model.train
    for epoch in range(args.epoch):
        trainer(args, logger=logger)

    model.save_model('latest_model.pth', save_path)

    logging.warning(f"GPU {args.rank} training is FINISHED")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='UnMixMatch')
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', type=str2bool, default=False)
    parser.add_argument('--use_tensorboard', action='store_true', help='Use tensorboard to plot and save curves, otherwise save the curves locally.')

    '''
    Method Specific 
    '''
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2 ** 20,
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=1000,
                        help='evaluation frequency')
    parser.add_argument('--num_labels', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='total number of batch size of labeled data')
    parser.add_argument('--uratio', type=int, default=1,
                        help='the ratio of unlabeled data to labelld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

    parser.add_argument('--alpha', type=float, default=0.75, help='param for Beta distribution of Mix Up')
    parser.add_argument('--T', type=float, default=0.5, help='Temperature Sharpening')
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--w_rot', type=float, default=0.5, help='weight for rot loss')
    parser.add_argument('--w_contrastive', type=float, default=0.1, help='weight for rot loss')
    parser.add_argument('--contrast_factor', type=str, default='strong', help='which to contrast between')
    parser.add_argument('--use_dm', type=str2bool, default=True, help='Whether to use distribution matching')
    parser.add_argument('--use_xe', type=str2bool, default=True, help='Whether to use cross-entropy or Brier')
    parser.add_argument('--warm_up', type=float, default=1 / 64)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', type=str2bool, default=True, help='use mixed precision training or not')
    parser.add_argument('--clip', type=float, default=0)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--dim_in', type=int, default=128)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--contrastive_temperature', type=float, default=0.5)
    parser.add_argument('--projection_head', type=str, default=None)
    parser.add_argument('--bn_momentum', type=float, default=0.001)

    '''
    Data Configurations
    '''
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--use_extra_data', type=str2bool, default=False,
                        help='whether or not use labeled data as extra data to unconstrained data')
    '''
    multi-GPUs & Distrbitued Training
    '''
    # args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    # config file
    parser.add_argument('--c', type=str, default='')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    main(args)
