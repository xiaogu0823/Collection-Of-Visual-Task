import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.utils_bbox import loc2bbox
from torchvision.ops import nms
from utils.anchors import generate_anchor_base, _enumerate_shifted_anchor


class ProposalCreator:
    """
    生成建议框 （超出边缘筛选、最小值、非极大值抑制）
    """

    def __init__(self,
                 mode,
                 nms_iou=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=600,
                 n_test_pre_nms=3000,
                 n_test_post_nms=300,
                 min_size=16):
        # 训练或者预测模式
        self.mode = mode
        self.nms_iou = nms_iou
        # 训练用到的建议框的数量
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        # 预测用到的建议框的数量
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        anchor = torch.from_numpy(anchor).type_as(loc)
        # 生成建议框proposal
        roi = loc2bbox(anchor, loc)
        # 防止建议框proposal超过图像边缘
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])
        # 建议框的最小值不得小于16
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))
        roi = roi[keep]
        # TODO score 是啥格式
        score = score[keep]

        # 根据得分进行排序，取出建议框
        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # 对建议框进行非极大值抑制（nms），此处使用官方
        keep = nms(roi, score, self.nms_iou)
        if len(keep) < n_post_nms:
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)))
            keep = torch.cat([keep, keep[index_extra]])
        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi


class RegionProposalNetwork(nn.Module):
    """
    RPN 网络：生成预测分类以及预测proposal的边框回归参数
    """

    def __init__(self,
                 in_channels=512,
                 mid_channels=512,
                 anchor_ratios=[0.5, 1, 2],
                 anchor_sizes=[128, 256, 512],
                 feature_stride=16,
                 mode="training"):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(anchor_ratios, anchor_sizes)
        n_anchor = self.anchor_base.shape[0]

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1)

        self.feature_stride = feature_stride
        self.proposal_layer = ProposalCreator(mode)

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape
        x = F.relu(self.conv1(x))
        rpn_locs = self.loc(x)
        # TODO .contiguous() 决定了内存的连续性
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)

        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)

        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feature_stride, h, w)
        rois = []
        roi_indices = []
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi.unsqueeze(0))
            roi_indices.append(batch_index.unsqueeze(0))

        rois = torch.cat(rois, dim=0).type_as(x)
        roi_indices = torch.cat(roi_indices, dim=0).type_as(x)
        anchor = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
