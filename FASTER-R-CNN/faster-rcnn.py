import cv2
import numpy as np
import torch

from backbone.vgg import VGG16
from backbone.rpn import RPNHead
from backbone.classifier import FinalNet
from bbox.anchors import generate_anchor_base, _enumerate_shifted_anchor
from bbox.offset import offset_inverse
from torchvision.ops import roi_pool

if __name__ == '__main__':
    # 将输入图片的尺寸，进行修改（将短边设置为600）
    image = cv2.imread("./img/catdog.jpg")
    h, w = image.shape[:2]
    current_h = 600
    current_w = int(600 / h * w)
    image = cv2.resize(image, (current_w, current_h))
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).permute([0, 3, 1, 2])

    # 进入主干网络
    device = torch.device("cuda:0")
    vgg = VGG16(3, device)
    features = vgg(image.to(device).float())

    # 根据尺度与比例生成anchors
    offsets = generate_anchor_base()
    anchors = _enumerate_shifted_anchor(offsets, 32, features.shape[-2], features.shape[-1])
    # print(anchors.shape) # 3888, 4

    # 进入RPN得到目标值（分类结果，边框回归参数）
    rpn = RPNHead()
    rpn = rpn.to(device)
    cls, reg = rpn(features)
    # print(cls.shape, reg.shape) # 3888, 2  3888, 4

    # 通过上一步的边框回归参数与anchors结合生成建议框（proposal）
    proposals = offset_inverse(anchors.to(device), reg)
    cls = torch.argmax(cls, dim=-1).reshape(-1, 1)
    proposals = torch.cat([cls, proposals], dim=1)

    # 经过roi_pool
    roi_pool_result = roi_pool(features, proposals, output_size=(7, 7))

    # 将roi——pool的结果进行全连接处理
    classifier = FinalNet(3)
    classifier = classifier.to(device)
    cls_predict, reg_predict = classifier(roi_pool_result)

    print(cls_predict.shape, reg_predict.shape)

    # 通过anchors获取labels
    # ............................. 此处仿照Fast RCNN 做法
