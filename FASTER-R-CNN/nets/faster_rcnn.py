from torch import nn
from nets.resnet50 import resnet50
from nets.rpn import RegionProposalNetwork
from nets.classifier import Resnet50RoIHead


class FasterRCNN(nn.Module):
    def __init__(self, num_classes,
                 mode="training",
                 feature_stride=16,
                 anchor_sizes=[128, 256, 512],
                 anchor_ratios=[0.5, 1, 2],
                 backbone='vgg',
                 pretrained=False):
        super(FasterRCNN, self).__init__()
        self.feature_stride = feature_stride
        self.extractor, classifier = resnet50()

        self.rpn = RegionProposalNetwork(
            1024, 1024,
            anchor_ratios=anchor_ratios,
            anchor_sizes=anchor_sizes,
            feature_stride=self.feature_stride,
            mode=mode
        )

        self.head = Resnet50RoIHead(
            n_class=num_classes + 1,
            roi_size=14,
            spatial_scale=1,
            classifier=classifier
        )

    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":
            img_size = x.shape[2:]
            base_feature = self.extractor.forward(x)
            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices

        elif mode == "extractor":
            base_feature = self.extractor.forward(x)
            return base_feature

        elif mode == "rpn":
            base_feature, img_size = x
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
