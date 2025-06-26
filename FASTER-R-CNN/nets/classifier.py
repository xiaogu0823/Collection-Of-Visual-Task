import torch
from torch import nn
from torchvision.ops import RoIPool


class Resnet50RoIHead(nn.Module):
    """
    针对建议框进行ROI pooling 并输出预测分类得分以及边框回归参数
    """

    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier

        self.cls_loc = nn.Linear(2048, n_class * 4)
        self.score = nn.Linear(2048, n_class)

        # 初始化两个层级的weights权重
        normal_init(self.cls_loc, 0, 0.01)
        normal_init(self.score, 0, 0.01)

        # ROI pooling 从 roi_size大小的方格中，生成roi_size的矩阵
        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        # roi_indices = roi_indices.cuda()
        # rois = rois.cuda()
        # 从两个维度展平维度
        rois = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        # 还原roi尺度到输入特征的形状
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]

        # 将分类下标，与边框回归参数进行拼接
        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        # 进行ROI pooling
        # TODO 猜测x是输入前的特征图大小，indices_and_rois 是对应的类别和边框回归参数
        pool = self.roi(x, indices_and_rois)

        fc7 = self.classifier(pool)
        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
