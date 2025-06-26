from torch import nn


class RPNHead(nn.Module):
    def __init__(self, num_anchors=9, num_classes=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
        )

        # 前景与背景的概率
        self.cls_fc = nn.Sequential(
            nn.Conv2d(512, num_anchors * num_classes, 1, 1),
            nn.Softmax(-1)
        )

        # 边框回归参数
        self.reg_fc = nn.Sequential(
            nn.Conv2d(512, num_anchors * 4, 1, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        cls_x = self.cls_fc(x)
        cls_x = cls_x.reshape(-1, 2)

        reg_x = self.reg_fc(x)
        reg_x = reg_x.reshape(-1, 4)
        # 类别预测参数、边框回归参数
        return cls_x, reg_x
