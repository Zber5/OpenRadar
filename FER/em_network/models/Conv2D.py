import torch
import torch.nn as nn
import torch.nn.functional as F
from FER.em_network.models.model import TimeDistributed


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


class ImageNet(nn.Module):
    def __init__(self, num_channel=1):
        super(ImageNet, self).__init__()

        self.group1 = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=(5, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)))

        self.group2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)))

        self.group3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        return x


class ImageDualNet(nn.Module):
    def __init__(self):
        super(ImageDualNet, self).__init__()

        self.azi_net = TimeDistributed(ImageNet())
        self.ele_net = TimeDistributed(ImageNet())

    def forward(self, azi, ele):
        out_azi = self.azi_net(azi)
        out_ele = self.ele_net(ele)
        out_azi = torch.mean(out_azi, dim=1)
        out_ele = torch.mean(out_ele, dim=1)
        out_azi = out_azi.view((out_azi.size(0), -1))
        out_ele = out_ele.view((out_ele.size(0), -1))
        return out_azi, out_ele


class ImageFull(nn.Module):
    def __init__(self, num_classes):
        super(ImageFull, self).__init__()

        self.azi_ele_net = ImageDualNet()
        self.fc1 = nn.Sequential(
            nn.Linear(1024 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, azi, ele):
        azi, ele = self.azi_ele_net(azi, ele)
        out = torch.cat((azi, ele), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class ImagePhaseNet(nn.Module):
    def __init__(self, num_classes):
        super(ImagePhaseNet, self).__init__()

        self.azi_ele_net = ImageDualNet()
        self.phase_net = PhaseNet()

        self.fc1 = nn.Sequential(
            nn.Linear(1024 * 2 + 1920, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, num_classes))

    def forward(self, azi, ele, phase):
        azi, ele = self.azi_ele_net(azi, ele)
        phase = self.phase_net(phase)
        phase = phase.view((phase.size(0), -1))
        out = torch.cat((azi, ele, phase), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


if __name__ == "__main__":
    # device = torch.device('cuda')
    # model = PhaseNet()
    # model = model.to(device)
    #
    # input1 = torch.randn(8, 12, 10, 100)
    # input1 = input1.to(device)
    # # output = model(input1)
    # output = model(input1)
    # print(output.view((output.size(0), -1)).size())

    device = torch.device('cuda')
    # model = TimeDistributed(ImageNet())
    # model = ImageDualNet()
    # model = ImageFull(num_classes=7)
    model = ImagePhaseNet(num_classes=7)
    model = model.to(device)

    input1 = torch.randn(8, 100, 1, 91, 10)
    input1 = input1.to(device)

    input2 = torch.randn(8, 100, 1, 91, 10)
    input2 = input2.to(device)

    input3 = torch.randn(8, 12, 10, 100)
    input3 = input3.to(device)
    # output = model(input1)
    # azi, ele = model(input1, input2)
    out = model(input1, input2, input3)
    print(out.size())
