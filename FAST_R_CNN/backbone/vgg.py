import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights
# 导入ROI pool
from torchvision.ops import roi_pool


class VGG16(nn.Module):
    def __init__(self, num_classes,device):
        super().__init__()
        self.device = device
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        vgg.to(device=device)
        self.features = vgg.features
        self.classifier = nn.Sequential(*list(vgg.classifier)[:-1])
        # 分类的全连接
        self.cls_fc = nn.Sequential(
            nn.Linear(4096, 1024,device=self.device),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes,device=self.device),
            nn.Softmax(dim=-1)
        )
        # 边框回归参数的全连接
        self.regressor_fc = nn.Sequential(
            nn.Linear(4096, 1024,device=self.device),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 4,device=self.device),
            nn.Sigmoid()
        )

        self.flatten = nn.Flatten()

    def forward(self, x, anchors):
        x = self.features(x)
        # ROIpool层
        x = roi_pool(x, anchors, (7, 7))
        x = self.classifier(self.flatten(x))
        x1 = self.cls_fc(x)
        x2 = self.regressor_fc(x)
        return x1, x2


if __name__ == '__main__':
    model = VGG16(2)

    anchors = torch.tensor([
        [0, 0.1, 0.2, 0.4, 0.5],
        [1, 0.2, 0.3, 0.3, 0.3]
    ])

    image = torch.randn(1, 3, 300, 500)
    cls, reg = model(image, anchors)
    print(cls, reg)
