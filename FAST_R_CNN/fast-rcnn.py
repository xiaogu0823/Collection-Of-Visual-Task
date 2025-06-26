import cv2
import numpy as np
import torch

from backbone.vgg import VGG16
from bbox.selective_search import generate_bbox
from bbox.nms import multibox_target, assign_anchor_to_bbox
from torch import nn

if __name__ == '__main__':
    # 实际中标注的框
    ground_truth = torch.tensor([[0.1, 0.08, 0.52, 0.92],
                                 [0.55, 0.2, 0.9, 0.88]])

    image = cv2.imread("./img/catdog.jpg")
    h, w = image.shape[:2]
    # --------------------ss算法生成anchors--------------------------
    # anchors 的位置 左上角坐标、框的宽高
    anchors = generate_bbox(image)
    anchors = torch.tensor(anchors) / torch.tensor([w, h, w, h])
    # ---------------------使用iou生成对应类别信息--------------------
    map = assign_anchor_to_bbox(ground_truth, anchors, device=torch.device("cpu"))
    map = map + 1  # 背景:0 狗：1，猫：2
    # 生成类别的labels
    cls_labels = map
    # 加入类别信息的anchors
    anchors = torch.concatenate([map.view(-1, 1), anchors], dim=-1)

    # 生成边框回归的labels
    reg_labels = []
    for anchor in anchors:
        c, px, py, pw, ph = anchor
        if c == 0:
            reg_labels.append([0., 0., 0., 0.])
        else:
            gx, gy, gw, gh = ground_truth[(c - 1).long()]
            dx = (gx - px) / pw
            dy = (gy - py) / ph
            dw = torch.log(gw / pw)
            dh = torch.log(gh / ph)
            reg_labels.append([dx.item(), dy.item(), dw.item(), dh.item()])
    reg_labels = torch.tensor(reg_labels)

    # 对输入特征参数进行调整（注意：图像以后再也不用调整大小了）
    features = np.expand_dims(image, 0)
    features = torch.from_numpy(features).permute([0, 3, 1, 2])

    # ---------------------------------训练-----------------------
    device = torch.device("cpu")
    model = VGG16(3, device)  # 背景 猫 狗
    features = features.to(device)
    anchors = anchors.to(device)
    cls_labels = cls_labels.to(device)
    reg_labels = reg_labels.to(device)

    # 联合损失率
    loss_cls = nn.CrossEntropyLoss()
    loss_reg = nn.SmoothL1Loss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        cls, reg = model(features.float(), anchors)
        # 联合损失率
        loss = loss_cls(cls, cls_labels.long()) + loss_reg(reg, reg_labels)
        loss.backward()
        optimizer.step()

        print(f"epoch {epoch + 1} / {epochs} -- loss: {loss.item():.4f}")
