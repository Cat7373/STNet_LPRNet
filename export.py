#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging

import torch

from model.lprnet import LPRNet, CHARS
from model.st import STNet
from utils.general import set_logging, MultiModelWrapper

logger = logging.getLogger(__name__)
set_logging()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opts = parser.parse_args()
    del parser

    # 打印参数
    logger.info("args: %s" % opts)

    # 预定义的参数(不打印)
    opts.img_size = (94, 24)
    opts.lpr_dropout_rate = .5

    # Input
    img = torch.zeros((opts.batch_size, 3, opts.img_size[1], opts.img_size[0]))

    # 定义网络
    device = torch.device('cpu')
    stnet = STNet()
    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=opts.lpr_dropout_rate)
    logger.info("Build network is successful.")

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

    # 组合模型
    model = MultiModelWrapper([stnet, lprnet])

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
    y = model(img)  # dry run

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opts.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'], output_names=['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('Export complete.')
