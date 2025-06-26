import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch


def draw_bbox(image, anchors):
    h, w = image.shape[:2]
    # 还原整个真实值的ground-truth
    # anchors[:, -4:] = anchors[:, -4:] * torch.tensor([w, h, w, h])
    # plt 绘制图像
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image)
    # plt 绘制真实框
    for anchor in anchors[:, -4:]:
        # x, y, w, h = anchor
        x1, y1, x2, y2 = anchor.int()
        w = x2 - x1
        h = y2 - y1
        r1 = mpatch.Rectangle((x1, y1), w, h, color="red", fill=False)
        ax.add_patch(r1)
    plt.show()


if __name__ == '__main__':
    # 加载图像
    image = cv2.imread("../img/catdog.jpg")
    # 真实框 （assigned-box）
    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                                 [1, 0.55, 0.2, 0.9, 0.88]])
    draw_bbox(image, ground_truth)

#
# # 还原整个真实值的ground-truth
# ground_truth[:, 1:] = ground_truth[:, 1:] * torch.tensor([w, h, w, h])
#
# # plt 绘制图像
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# ax.imshow(image)
# # plt 绘制真实框
# for gt in ground_truth:
#     x1, y1, x2, y2 = gt[1:]
#     w = x2 - x1
#     h = y2 - y1
#     r1 = mpatch.Rectangle((x1, y1), w, h, color="red", fill=False)
#     ax.add_patch(r1)
#
# plt.show()
