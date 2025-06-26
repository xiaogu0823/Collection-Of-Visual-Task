import torch


def corner2center(bbox):
    """
    将左上角坐标与右下角坐标 转化为 中心点坐标与宽高模式 此处考虑（batch_size anchors_num 4）
    :param bbox: 先验框或者建议框
    :return:
    """
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    return torch.stack([cx, cy, w, h], dim=-1)


def center2corner(bbox):
    """
        将中心点坐标与宽 转化为 左上角坐标与右下角坐标高模式 此处考虑（batch_size anchors_num 4）
        :param bbox: 先验框或者建议框
        :return:
        """
    cx = bbox[:, 0]
    cy = bbox[:, 1]
    w = bbox[:, 2]
    h = bbox[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=-1)
