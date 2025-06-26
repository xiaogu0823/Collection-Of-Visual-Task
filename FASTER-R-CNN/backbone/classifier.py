from torch import nn


class FinalNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU()
        )

        self.cls_fc = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Softmax(-1)
        )
        self.reg_fc = nn.Sequential(
            nn.Linear(1024, 4)
        )

    def forward(self, x):
        x = self.fc(self.flatten(x))
        cls_x = self.cls_fc(x)
        reg_x = self.reg_fc(x)
        return cls_x, reg_x
