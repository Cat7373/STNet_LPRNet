#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from enum import Enum, unique
from typing import List, Dict

from utils.general import load_cache, save_cache


@unique
class ImgType(Enum):
    """
    图片类型
    元素为: (存储目录, 描述)
    """
    base = ("ccpd_base", "正常车牌")
    challenge = ("ccpd_challenge", "比较有挑战性的车牌")
    db = ("ccpd_db", "光线较暗或较亮")
    fn = ("ccpd_fn", "距离摄像头较远或较近")
    np = ("ccpd_np", "没上牌的新车")
    rotate = ("ccpd_rotate", "水平倾斜20-50°，垂直倾斜-10-10°")
    tilt = ("ccpd_tilt", "水平倾斜15-45°，垂直倾斜15-45°")
    weather = ("ccpd_weather", "雨天、雪天或者雾天的车牌")


class Img(object):
    def __init__(self, src: str, img_type: ImgType, label: str = None, rect: [int] = None):
        """
        :param src: 图片路径
        :param img_type: 图片类型
        :param label: 正确的车牌号
        :param rect: [x, y, x, y], 用于标记车牌位置
        """
        self.src = src
        self.img_type = img_type
        self.label = label
        self.rect = rect


regions: List[str] = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
codes: List[str] = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
imgs: Dict[ImgType, List[Img]] = {}


"""
文件名说明：
025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
由分隔符'-'分为几个部分:
1) 025为区域
2) 95_113 对应两个角度, 水平95°, 竖直113°
3) 154&383_386&473对应边界框坐标:左上(154, 383), 右下(386, 473)
4) 386&473_177&454_154&383_363&402对应四个角点坐标
5) 0_0_22_27_27_33_16为车牌号码 映射关系如下: 第一个为省份0 对应省份字典皖, 后面的为字母和文字, 查看ads字典.如0为A, 22为Y....
6) 37 为亮度
7) 15 为模糊度
"""

# 00249042145593-91_86-330&463_433&507-422&499_333&500_332&470_421&469-0_0_2_30_25_29_7-94-18.jpg
# 区域: 00249042145593
# 角度: 水平 91, 垂直 86
# 车牌范围: (y: 330, x: 463), (y: 433, x: 507)
# 车牌角点(非矩形范围): (y: 422, x: 499), (y: 333, x: 500), (y: 332, x: 470), (y: 421, x: 469)
# 车牌号: 0_0_2_30_25_29_7
# 亮度: 94
# 模糊度: 18


def load_img(img_path: str, file_name: str, img_type: ImgType):
    # 去后缀
    img_name, suffix = os.path.splitext(file_name)

    # 按 - 切分
    img_name_split = img_name.split('-')
    if len(img_name_split) < 5:
        return

    # 获得车牌部分
    number = img_name_split[4]
    pre_label = number.split('_')
    if len(pre_label) < 3:
        return
    label = regions[int(pre_label[0])]
    for i in range(1, len(pre_label)):
        label += codes[int(pre_label[i])]

    # rect
    rect_split = img_name_split[2].split("_")
    rect_x1y1 = rect_split[0].split("&")
    rect_x2y2 = rect_split[1].split("&")

    rect = (int(rect_x1y1[0]), int(rect_x1y1[1]), int(rect_x2y2[0]), int(rect_x2y2[1]))

    return Img(img_path, img_type, label, rect)


def load_ccpd_imgs(source_dir):
    # from cache
    cached_imgs = load_cache('cache_sourceset.pt')
    if cached_imgs:
        if cached_imgs['source_dir'] == source_dir:
            for (k, v) in cached_imgs['imgs'].items():
                imgs[k] = v
            return

    for img_type in ImgType:
        folder_name = img_type.value[0]
        folder_path = os.path.join(source_dir, folder_name)

        img_list = []
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if img_type == ImgType.np:
                img = Img(filepath, img_type)
                img_list.append(img)
            else:
                img = load_img(filepath, filename, img_type)
                img_list.append(img)
        imgs[img_type] = img_list

    save_cache('cache_sourceset.pt', {
        'source_dir': source_dir,
        'imgs': imgs
    })
