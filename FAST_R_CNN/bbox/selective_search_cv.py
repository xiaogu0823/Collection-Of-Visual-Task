import sys
import cv2
import torch

from draw.draw_bbox import draw_bbox


def generate_bbox(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)

    # 可以选择快速但是低 recall 的方式
    # 这里的 recall 指的是选择出来的 region 是否包含了所有应该包含的区域。recall 越高越好
    # ss.switchToSelectiveSearchFast()
    # 也可以选择慢速但是高 recall 的方式
    ss.switchToSelectiveSearchQuality()
    # rects： x,y 左上角坐标 w,h 宽度与高度
    rects = ss.process()
    bboxes = []
    for rect in rects:
        x, y, w, h = rect
        if w < 100:
            continue
        if h < 100:
            continue
        if w * h < 500:
            continue
        bboxes.append(rect)
    return bboxes[:500]


if __name__ == '__main__':
    # 此处选择性搜索算法的颜色标准是RGB
    image = cv2.imread("../img/catdog.jpg")
    h, w = image.shape[:2]
    anchors = generate_bbox(image)
    anchors = torch.tensor(list(anchors)) / torch.tensor([w, h, w, h])
    draw_bbox(image, anchors)
