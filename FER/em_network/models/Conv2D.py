import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseNet(nn.Module):
    def __init__(self):
        super(PhaseNet, self).__init__()

        self.group1 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=(5, 5), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)))

        self.group2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=(3, 5), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)))

        self.group3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=(5, 5), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
        )

    def forward(self, x):
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        return x


if __name__ == "__main__":
    device = torch.device('cuda')
    model = PhaseNet()
    model = model.to(device)

    input1 = torch.randn(8, 12, 10, 100)
    input1 = input1.to(device)
    # output = model(input1)
    output = model(input1)
    print(output.size())
