import torch
from torchvision.models import vgg16, VGG16_Weights
from torch import nn


class VGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.vgg_conv = nn.Sequential(*list(vgg.features)[:22])
        self.vgg_conv[4] = nn.Conv2d(64, 64, 3, 2, 1)
        self.vgg_conv[9] = nn.Conv2d(128, 128, 3, 2, 1)
        self.vgg_conv[16] = nn.Conv2d(256, 256, 3, 2, 1)

        self.conv_6 = nn.Conv2d(512, 1024, 3, 2, 1)
        self.conv_7 = nn.Conv2d(1024, 1024, 1, 1)

        self.conv_8_1 = nn.Conv2d(1024, 256, 1, 1)
        self.conv_8_2 = nn.Conv2d(256, 512, 3, 2, 1)

        self.conv_9_1 = nn.Conv2d(512, 128, 1, 1)
        self.conv_9_2 = nn.Conv2d(128, 256, 3, 2, 1)

        self.conv_10_1 = nn.Conv2d(256, 128, 1, 1)
        self.conv_10_2 = nn.Conv2d(128, 256, 3, 2, 1)

        self.conv_11_1 = nn.Conv2d(256, 128, 1, 1)
        self.conv_11_2 = nn.Conv2d(128, 256, 3, 1)

        self.generate_1 = nn.Conv2d(512, 4 * (num_classes + 4), 3, 1, 1)
        self.generate_2 = nn.Conv2d(1024, 6 * (num_classes + 4), 3, 1, 1)
        self.generate_3 = nn.Conv2d(512, 6 * (num_classes + 4), 3, 1, 1)
        self.generate_4 = nn.Conv2d(256, 6 * (num_classes + 4), 3, 1, 1)
        self.generate_5 = nn.Conv2d(256, 4 * (num_classes + 4), 3, 1, 1)
        self.generate_6 = nn.Conv2d(256, 4 * (num_classes + 4), 3, 1, 1)

    def forward(self, x):
        x = self.vgg_conv(x)
        c1 = self.generate_1(x)

        x = self.conv_6(x)
        x = self.conv_7(x)
        c2 = self.generate_2(x)

        x = self.conv_8_1(x)
        x = self.conv_8_2(x)
        c3 = self.generate_3(x)

        x = self.conv_9_1(x)
        x = self.conv_9_2(x)
        c4 = self.generate_4(x)

        x = self.conv_10_1(x)
        x = self.conv_10_2(x)
        c5 = self.generate_5(x)

        x = self.conv_11_1(x)
        x = self.conv_11_2(x)

        c6 = self.generate_6(x)

        c1 = c1.reshape(-1, self.num_classes + 4)
        c2 = c2.reshape(-1, self.num_classes + 4)
        c3 = c3.reshape(-1, self.num_classes + 4)
        c4 = c4.reshape(-1, self.num_classes + 4)
        c5 = c5.reshape(-1, self.num_classes + 4)
        c6 = c6.reshape(-1, self.num_classes + 4)
        c = torch.cat([c1, c2, c3, c4, c5, c6], dim=0)
        return c


if __name__ == '__main__':
    model = VGG(2)
    image = torch.randn(1, 3, 300, 300)
    result = model(image)
    print(result.shape)
