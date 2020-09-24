#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import LPRDataSet, collate_fn
from data.ccpd import load_ccpd_imgs
from model.lprnet import LPRNet, CHARS
from model.st import STNet
from utils.general import decode, sparse_tuple_for_ctc, set_logging, set_cache_dir

logger = logging.getLogger(__name__)
set_logging()


def test(lprnet, st, data_loader, dataset, device, ctc_loss, lpr_max_len, float_test=False):
    correct_count = 0
    process_count = 0

    half = not float_test and (device.type != 'cpu')
    if half:
        st.half()
        lprnet.half()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc='Test')
    mloss = .0
    for i, (imgs, labels, lengths) in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        imgs = imgs.half() if half else imgs.float()
        labels = labels.half() if half else labels.float()

        # 泛化
        imgs -= 127.5
        imgs *= .0078431  # 127.5 * 0.0078431 = 0.99999525

        # 随机底片
        if random.random() > .5:
            imgs = -imgs

        # 准备 loss 计算的参数
        input_lengths, target_lengths = sparse_tuple_for_ctc(lpr_max_len, lengths)

        with torch.no_grad():
            x = st(imgs)
            x = lprnet(x)
            y = x.permute(2, 0, 1)  # [batch_size, chars, width] -> [width, batch_size, chars]
            y = y.log_softmax(2).requires_grad_()
            loss = ctc_loss(y.float(), labels.float(), input_lengths=input_lengths, target_lengths=target_lengths)

        x = x.cpu().detach().numpy()
        _, pred_labels = decode(x)

        start = 0
        for j, length in enumerate(lengths):
            label = labels[start:start + length]
            start += length
            if np.array_equal(np.array(pred_labels[j]), label.cpu().numpy()):
                correct_count += 1

        # Print
        mloss = (mloss * i + loss.item()) / (i + 1)  # update mean losses
        process_count += len(lengths)
        acc = float(correct_count) / float(process_count)
        pbar.set_description('Test mloss: %.5f, macc: %.5f' % (mloss, acc))

    acc = float(correct_count) / float(len(dataset))

    st.float()
    lprnet.float()

    return mloss, acc


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
