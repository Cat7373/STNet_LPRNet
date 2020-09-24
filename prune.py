#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os

import torch
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from data.dataset import LPRDataSet, collate_fn
from data.ccpd import load_ccpd_imgs
from model.lprnet import LPRNet, CHARS
from model.st import STNet
from test import test
from utils.general import set_logging, set_cache_dir

logger = logging.getLogger(__name__)
set_logging()


def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune_model(model, rate, model_name):
    if rate == 0.0:
        return

    parameters_to_prune = []
    for name, module in model.named_modules():
        try:
            _ = module.weight
        except:
            continue

        parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=rate,
    )  # prune
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')  # make permanent

    logger.info('  %s %.3f%% sparsity' % (model_name, sparsity(model) * 100))


def main(opts):
    # 选择设备
    device = torch.device("cuda:0" if (not opts.cpu and torch.cuda.is_available()) else "cpu")
    cuda = device.type != 'cpu'
    logger.info('Use device %s.' % device)

    # 定义网络
    stnet = STNet()
    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=opts.lpr_dropout_rate)
    stnet, lprnet = stnet.to(device), lprnet.to(device)
    logger.info("Build network is successful.")

    # 损失函数
    ctc_loss = torch.nn.CTCLoss(blank=len(CHARS) - 1, reduction='mean')  # reduction: 'none' | 'mean' | 'sum'

    # Load weights
    ckpt = torch.load(opts.weights, map_location=device)
    epoch = ckpt['epoch']

    # 加载网络
    if 'stn' in ckpt:  # 兼容旧的保存格式
        stnet.load_state_dict(ckpt["stn"])
    else:
        stnet.load_state_dict(ckpt["st"])
    lprnet.load_state_dict(ckpt["lpr"])

    # 释放内存
    del ckpt

    # Print
    logger.info('Load weights completed.')

    # 加载数据
    test_dataset = LPRDataSet(opts.img_size, .8501, .15, cache=opts.cache_images)
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers,
                             pin_memory=cuda, collate_fn=collate_fn)

    logger.info('Image sizes %d test' % (len(test_dataset)))
    logger.info('Using %d dataloader workers' % opts.workers)

    # 剪枝前测试
    stnet.eval()
    lprnet.eval()
    test(lprnet, stnet, test_loader, test_dataset, device, ctc_loss, opts.lpr_max_len, opts.float_test)
    stnet.train()
    lprnet.train()

    # Purge
    logger.info('Pruning model... ')
    prune_model(stnet, opts.st_prune_rate, 'st')
    prune_model(lprnet, opts.lpr_prune_rate, 'lpr')

    # 保存结果
    torch.save({
        "epoch": epoch,
        "lpr": lprnet.state_dict(),
        "st": stnet.state_dict()
    }, opts.weights + '.prune.pt')

    # 剪枝后测试
    stnet.eval()
    lprnet.eval()
    test(lprnet, stnet, test_loader, test_dataset, device, ctc_loss, opts.lpr_max_len, opts.float_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STNet & LPRNet Testing')
    parser.add_argument('--source-dir',       type=str,            required=True,     help='train images source dir.')
    parser.add_argument('--weights',          type=str,            required=True,     help='initial weights path.')

    parser.add_argument('--cpu',              action='store_true',                    help='force use cpu.')
    parser.add_argument('--batch-size',       type=int,            default=128,       help='train batch size.')
    parser.add_argument('--float-test',       action='store_true',                    help='use float model run test.')

    parser.add_argument('--workers',          type=int,            default=-1,        help='maximum number of dataloader workers.')
    parser.add_argument('--cache-images',     action='store_true',                    help='cache images for faster test.')

    parser.add_argument('--worker-dir',       type=str,            default='runs',    help='worker dir.')

    parser.add_argument('--st-prune-rate',    type=float,          default=.3,        help='')
    parser.add_argument('--lpr-prune-rate',   type=float,          default=.3,        help='')

    args = parser.parse_args()
    del parser

    # 自动调整的参数
    if args.workers < 0:
        if args.cache_images:
            args.workers = 1
        else:
            args.workers = os.cpu_count()
    args.workers = min(os.cpu_count(), args.workers)

    # 打印参数
    logger.info("args: %s" % args)

    # 预定义的参数(不打印)
    args.img_size = (94, 24)
    args.lpr_max_len = 9 * 2  # 车牌最大位数 * 2
    args.lpr_dropout_rate = .5

    # 自动调整的参数(不打印)
    args.cache_dir = os.path.join(args.worker_dir, 'cache')

    # 参数处理后的初始化工作
    os.makedirs(args.cache_dir, exist_ok=True)
    set_cache_dir(args.cache_dir)
    load_ccpd_imgs(args.source_dir)

    # 开始训练
    main(args)
