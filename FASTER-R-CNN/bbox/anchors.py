"""
对应feature map 生成anchors
"""
import torch
import numpy as np
from draw.draw_bbox import draw_bbox


def generate_anchor_base(anchors_sizes=(64, 128, 256), anchors_ratios=(0.5, 1., 2.)):
    anchors_offsets = []
    for anchors_size in anchors_sizes:
        for anchors_ratio in anchors_ratios:
            h = (anchors_size ** 2 / anchors_ratio) ** 0.5
            w = h * anchors_ratio
            anchors_offsets.append([-w, -h, w, h])
    return torch.tensor(anchors_offsets) / 2


def _enumerate_shifted_anchor(offsets, stride, width, height):
    """
    生成原图上所有对应特征位置的anchors
    :param offsets: generate_anchor_base 每个中心点坐标生成的9个anchors偏移量
    :param stride: 卷积之后的小图映射到大图的步长（缩放的倍率大小）
    :param width:  卷积后的特征图的宽度
    :param height: 卷积后的特征图的高度
    :return:
    """
    anchors = []
    for i in range(width):
        for j in range(height):
            img_x = i * stride
            img_y = j * stride
            loc_anchors = offsets + torch.tensor([img_x, img_y, img_x, img_y])
            # 限制anchors不可以越界
            loc_anchors[:, :2] = torch.clamp(loc_anchors[:, :2], min=0)
            loc_anchors[:, 2] = torch.clamp(loc_anchors[:, 2], max=width * stride)
            loc_anchors[:, 3] = torch.clamp(loc_anchors[:, 3], max=height * stride)
            anchors.append(loc_anchors)

    anchors = torch.stack(anchors, dim=0).view(-1, 4)
    return anchors


if __name__ == '__main__':
    anchors_offsets = generate_anchor_base()
    anchors = _enumerate_shifted_anchor(anchors_offsets, 32, 18, 18)
    # print(anchors)

    image = np.ones((32 * 18, 32 * 18), dtype=np.uint8) * 255
    draw_bbox(image, anchors[[1118, 1119, 1120, 1121, 1122, 1123, 1124], :])
