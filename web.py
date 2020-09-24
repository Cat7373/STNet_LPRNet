# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import logging
import time

import numpy as np
import cv2
from flask import Flask, request
import torch

from utils.general import decode
from utils.web_common import fail, success

app = Flask('ML')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JSON_AS_ASCII'] = False
model = device = None


def build_model(opts):
    global model, device

    from model.lprnet import LPRNet, CHARS
    from model.st import STNet
    from utils.general import MultiModelWrapper

    device = torch.device("cuda:0" if (not opts.cpu and torch.cuda.is_available()) else "cpu")

    stnet = STNet()
    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)

    ckpt = torch.load(opts.weights, map_location=device)
    if 'stn' in ckpt:  # 兼容旧的保存格式
        stnet.load_state_dict(ckpt["stn"])
    else:
        stnet.load_state_dict(ckpt["st"])
    lprnet.load_state_dict(ckpt["lpr"])
    del ckpt

    stnet, lprnet = stnet.to(device), lprnet.to(device)
    stnet.eval()
    lprnet.eval()

    model = MultiModelWrapper([stnet, lprnet])


def eval(image, rect):
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]
    img_box = image[y1:y2 + 1, x1:x2 + 1, :]

    im = cv2.resize(img_box, (94, 24), interpolation=cv2.INTER_CUBIC)
    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * .0078431
    data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94])

    since = time.time()
    with torch.no_grad():
        result = model(data)
    used_time = time.time() - since

    result = result.cpu().detach().numpy()
    labels, _ = decode(result)

    return labels[0], used_time


@app.route("/api/ml/st_lpr", methods=["POST"])
def st_lpr() -> dict:
    if 'img' not in request.files:
        return fail('required param img is empty.')

    img = request.files['img']

    with img.stream as f:
        data = f.read()

    img_array = np.asarray(bytearray(data), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if (not ('rect' in request.form)) or (request.form['rect'] == ''):
        rect = [0, 0, image.shape[1], image.shape[0]]
    else:
        rect = list([int(x) for x in request.form['rect'].split(',')])
        rect = rect[:len(rect) // 4 * 4]

    labels = []
    total_used_time = 0.0
    for i in range(0, len(rect), 4):
        label, used_time = eval(image, rect[i:i + 4])
        if len(label) < 7 or len(label) > 8:
            labels.append('')
        else:
            labels.append(label)
        total_used_time += used_time

    return success(labels, msg='model used %.4fs.' % total_used_time)


def main(opts):
    build_model(opts)
    app.run(port=opts.port, host='0.0.0.0' if opts.listen_all else None)  # 注意，阻塞性操作


if __name__ == "__main__":
    # logger
    os.makedirs("runs/api", exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s]: %(message)s",
                        datefmt="%y-%m-%d %H:%M:%S",
                        filename="runs/api/%s.log" % time.strftime("%y-%m-%d_%H_%M_%S"),
                        filemode="w")
    sh = logging.StreamHandler()
    sh.setFormatter(logging.getLogger().handlers[0].formatter)
    logging.getLogger().addHandler(sh)

    # params
    parser = argparse.ArgumentParser(description='STNet & LPRNet web interface')
    parser.add_argument('--weights',    type=str,            required=True, help='weights file path.')
    parser.add_argument('--port',       type=int,            default=8088,  help='listening port.')
    parser.add_argument('--listen-all', action='store_true',                help='listening all interface.')
    parser.add_argument('--cpu',        action='store_true',                help='force use cpu.')

    args = parser.parse_args()
    del parser

    main(args)
