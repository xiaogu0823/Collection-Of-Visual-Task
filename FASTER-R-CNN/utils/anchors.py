import numpy as np


def generate_anchor_base(anchor_ratios=[0.5, 1, 2], anchor_sizes=[128, 256, 512]):
    """
    生成以0，0为中心的，先验框anchors 左上与右下的偏移量
    :param anchor_ratios: anchors的不同比例
    :param anchor_sizes: anchors的不同大小
    :return:
    """
    anchor_base = np.zeros((len(anchor_ratios) * len(anchor_sizes), 4), dtype=np.float32)
    for i in range(len(anchor_ratios)):
        for j in range(len(anchor_sizes)):
            h = anchor_sizes[j] / anchor_ratios[i] ** 0.5
            w = h * anchor_ratios[i]

            index = i * len(anchor_sizes) + j
            anchor_base[index] = np.array([w, h, w, h]) / 2.
    return anchor_base


def _enumerate_shifted_anchor(anchor_base, feature_stride, height, width):
    """

    :param anchor_base: 4坐标对称偏移量
    :param feature_stride: 缩放步长（resnet50的缩放倍率为16）
    :param height: backbone后得到的feature高度
    :param width: backbone后得到的feature宽度
    :return: 生成的先验框anchors
    """
    shift_x = np.arange(0, feature_stride * width, feature_stride)
    shift_y = np.arange(0, feature_stride * height, feature_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y, indexing='ij')
    shift = np.stack([shift_x, shift_y, shift_x, shift_y], axis=0)

    anchor = anchor_base.reshape((1, -1, 4)) + shift.reshape((-1, 1, 4))
    anchor = anchor.reshape((-1, 4)).astype(np.float32)
    return anchor
