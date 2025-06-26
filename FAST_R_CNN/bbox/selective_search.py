"""
selectivesearch 选择性搜索的API
"""
import cv2
import numpy as np
import torch
from selectivesearch import selective_search
from draw.draw_bbox import draw_bbox

scale = 100
sigma = 0.9
min_size = 20


def generate_bbox(image):
    ss_result = selective_search(image, scale, sigma, min_size)
    rect_list = []
    rects = set()
    for item in ss_result[1]:
        # 针对先验框w或h进行筛选
        w = item['rect'][2] - item['rect'][0]
        h = item['rect'][3] - item['rect'][1]
        if w < 100:
            continue
        if h < 100:
            continue
        # 针对先验框的面积大小进行筛选
        s = item['size']
        if s < 500:
            continue
        # 去掉重复的区域的筛选
        if item['rect'] in rects:
            continue
        rects.add(item['rect'])
        rect_list.append(item['rect'])

    return rect_list


if __name__ == '__main__':
    # 此处选择性搜索算法的颜色标准是RGB
    image = cv2.imread("../img/catdog.jpg")
    h, w = image.shape[:2]
    anchors = generate_bbox(image)
    print(len(anchors))
    anchors = torch.tensor(anchors) / torch.tensor([w, h, w, h])
    draw_bbox(image, anchors)
