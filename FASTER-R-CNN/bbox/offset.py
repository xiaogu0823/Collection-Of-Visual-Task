from bbox.utils import corner2center, center2corner
import torch


def offset_boxes(anchors, gts, eps=1e-6):
    c_anchors = corner2center(anchors)
    c_gts = corner2center(gts)
    offset_xy = (c_gts[:, :2] - c_anchors[:, :2]) / c_anchors[:, 2:]
    offset_wh = torch.log(eps + c_gts[:, 2:] / c_anchors[:, 2:])
    return torch.cat([offset_xy, offset_wh], dim=1)


def offset_inverse(anchors, target_offset):
    c_anchors = corner2center(anchors)
    c_predict_xy = c_anchors[:, 2:] * target_offset[:, :2] + c_anchors[:, :2]
    c_predict_wh = c_anchors[:, 2:] * torch.exp(target_offset[:, 2:])
    c_predict = torch.cat([c_predict_xy, c_predict_wh], dim=1)
    return center2corner(c_predict)
