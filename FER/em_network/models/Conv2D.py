import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from FER.em_network.models.model import TimeDistributed
from FER.em_network.models.LSTM import PhaseLSTM
from FER.em_network.models.Attention import SpatialGate_v1
from FER.em_network.models.resnet import ResNet10


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


class ImageNet_Large_v1(nn.Module):
    def __init__(self, num_channel=3):
        super(ImageNet_Large_v1, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.group1(x)
        g1 = self.group2(x)
        g2 = self.group3(g1)
        g3 = self.group4(g2)
        g4 = self.group5(g3)
        # x = self.avgpool(x)
        return g1, g2, g3, g4


class ImageNet_Large(nn.Module):
    def __init__(self, num_channel=1):
        super(ImageNet_Large, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        # x = self.avgpool(x)
        return x


class ImageNet_Square(nn.Module):
    def __init__(self, num_channel=1):
        super(ImageNet_Square, self).__init__()

        self.group1 = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=(5, 5), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)))

        self.group2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)))

        self.group3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
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


class ImageNet_Attention(nn.Module):
    def __init__(self, num_channel=1):
        super(ImageNet_Attention, self).__init__()

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

        self.att1 = SpatialGate_v1((5, 3))

        self.group4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )

        self.att2 = SpatialGate_v1((3, 2))

        self.group5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 1)),
        )

        self.att3 = SpatialGate_v1((3, 2))

    def forward(self, x):
        x = self.group1(x)  # 44,10
        x = self.group2(x)  # 21, 5
        x = self.group3(x)  # 9, 5
        x = self.att1(x)
        x = self.group4(x)  # 4, 2
        x = self.att2(x)
        x = self.group5(x)  # 4, 2
        x = self.att3(x)
        return x


class ImageNet_Attention_v1(nn.Module):
    def __init__(self, num_channel=1):
        super(ImageNet_Attention_v1, self).__init__()

        self.group1 = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=(5, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)))

        self.att1 = SpatialGate_v1((7, 3))

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

        self.att2 = SpatialGate_v1((3, 2))

        self.group5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.group1(x)
        x = self.att1(x)  # 44,10
        x = self.group2(x)  # 21, 5
        x = self.group3(x)  # 9, 5
        x = self.group4(x)  # 4, 2
        x = self.att2(x)
        x = self.group5(x)  # 4, 2

        return x


class ImageDualNet(nn.Module):
    def __init__(self, block=ImageNet):
        super(ImageDualNet, self).__init__()

        self.azi_net = TimeDistributed(block())
        self.ele_net = TimeDistributed(block())

    def forward(self, azi, ele):
        out_azi = self.azi_net(azi)
        out_ele = self.ele_net(ele)
        out_azi = torch.mean(out_azi, dim=1)
        out_ele = torch.mean(out_ele, dim=1)
        out_azi = out_azi.view((out_azi.size(0), -1))
        out_ele = out_ele.view((out_ele.size(0), -1))
        return out_azi, out_ele


class ImageFull_Square(nn.Module):
    def __init__(self, num_classes, block=ImageDualNet, subblock=ImageNet_Square):
        super(ImageFull_Square, self).__init__()

        self.azi_ele_net = block(block=subblock)
        # self.fc1 = nn.Sequential(
        #     nn.Linear(128 * 3 * 3 * 2, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )

        self.fc1 = nn.Sequential(
            nn.Linear(2048 * 2 * 9, 512),
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


class ImageDualNet_Single(nn.Module):
    def __init__(self, block=ImageNet):
        super(ImageDualNet_Single, self).__init__()

        self.azi_net = block()
        self.ele_net = block()

    def forward(self, azi, ele):
        out_azi = self.azi_net(azi)
        out_ele = self.ele_net(ele)
        out_azi = out_azi.view((out_azi.size(0), -1))
        out_ele = out_ele.view((out_ele.size(0), -1))
        return out_azi, out_ele


class ImageDualNet_Single_v1(nn.Module):
    def __init__(self, block=ImageNet):
        super(ImageDualNet_Single_v1, self).__init__()

        self.azi_net = block()
        self.ele_net = block()

    def forward(self, azi, ele):
        g1_azi, g2_azi, g3_azi, g4_azi = self.azi_net(azi)
        g1_ele, g2_ele, g3_ele, g4_ele = self.ele_net(ele)
        g1 = torch.cat((g1_azi, g1_ele), dim=1)
        g2 = torch.cat((g2_azi, g2_ele), dim=1)
        g3 = torch.cat((g3_azi, g3_ele), dim=1)
        g4 = torch.cat((g4_azi, g4_ele), dim=1)
        out_azi = g4_azi.view((g4_azi.size(0), -1))
        out_ele = g4_ele.view((g4_ele.size(0), -1))
        return out_azi, out_ele, (g1, g2, g3, g4)


class ImageDualNet_Attention_Single(nn.Module):
    def __init__(self, block=ImageNet_Attention):
        super(ImageDualNet_Attention_Single, self).__init__()

        self.azi_net = block()
        self.ele_net = block()

    def forward(self, azi, ele):
        out_azi = self.azi_net(azi)
        out_ele = self.ele_net(ele)
        out_azi = out_azi.view((out_azi.size(0), -1))
        out_ele = out_ele.view((out_ele.size(0), -1))
        return out_azi, out_ele


class ImageFull(nn.Module):
    def __init__(self, num_classes, block=ImageNet, subblock=ImageNet_Attention):
        super(ImageFull, self).__init__()

        self.azi_ele_net = block(block=subblock)
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


class ImageSingle(nn.Module):
    def __init__(self, num_classes, block=ImageDualNet_Single, subblock=ImageNet_Large):
        super(ImageSingle, self).__init__()

        self.azi_ele_net = block(block=subblock)
        self.fc1 = nn.Sequential(
            nn.Linear(1024 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # self.fc1 = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )

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


class Classifier_Transformer(nn.Module):
    def __init__(self, input_dim=512, num_classes=7):
        super(Classifier_Transformer, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class ImageSingle_v1(nn.Module):
    def __init__(self, num_classes, block=ImageDualNet_Single, subblock=ImageNet_Large):
        super(ImageSingle_v1, self).__init__()

        self.azi_ele_net = block(block=subblock)
        self.fc1 = nn.Sequential(
            nn.Linear(1024 * 2 * 2, 512),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

        # self.fc1 = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, azi, ele):
        azi, ele, g = self.azi_ele_net(azi, ele)
        out = torch.cat((azi, ele), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out, g


class ImageSingle_Student(nn.Module):
    def __init__(self, num_classes):
        super(ImageSingle_Student, self).__init__()

        self.azi_ele_net = ImageDualNet_Single()
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
        out1 = self.fc2(out)
        out2 = self.fc3(out1)
        return out, out2


class ImageSingle_Attention(nn.Module):
    def __init__(self, num_classes):
        super(ImageSingle_Attention, self).__init__()

        self.azi_ele_net = ImageDualNet_Attention_Single()
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


class ImagePhaseNet_Single(nn.Module):
    def __init__(self, num_classes):
        super(ImagePhaseNet_Single, self).__init__()

        self.azi_ele_net = ImageDualNet_Single()
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


class ImagePhaseLSTM_Single(nn.Module):
    def __init__(self, input_size=12, hidden_size=256, batch_size=8, num_classes=7):
        super(ImagePhaseLSTM_Single, self).__init__()

        self.azi_ele_net = ImageDualNet_Single()
        self.phase_net = PhaseLSTM(input_size=input_size, hidden_size=hidden_size, batch_size=batch_size)

        self.fc1 = nn.Sequential(
            nn.Linear(1024 * 2 + hidden_size, 512),
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
        out = torch.cat((azi, ele, phase), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class ImagePhaseLSTM_Single_v1(nn.Module):
    def __init__(self, input_size=12, hidden_size=256, batch_size=8, num_classes=7):
        super(ImagePhaseLSTM_Single_v1, self).__init__()

        self.azi_ele_net = ImageDualNet_Single()
        self.phase_net = PhaseLSTM(input_size=input_size, hidden_size=hidden_size, batch_size=batch_size)
        self.relu = nn.ReLU()

        self.azi_ele_fc = nn.Sequential(
            nn.Linear(1024 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512 + hidden_size, 512),
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
        phase = self.relu(phase)
        azi_ele = torch.cat((azi, ele), dim=1)
        azi_ele = self.azi_ele_fc(azi_ele)
        out = torch.cat((azi_ele, phase), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class TFblock_v3(nn.Module):
    def __init__(self, dim_in=512, dim_inter=1024):
        super(TFblock_v3, self).__init__()
        # self.encoder = nn.Linear(dim_in, dim_inter)
        # self.decoder = nn.Linear(dim_inter, dim_in)

        self.encoder = nn.Conv2d(in_channels=dim_in, out_channels=dim_inter, kernel_size=(4, 2))
        self.bn1 = nn.BatchNorm2d(num_features=dim_inter)
        self.decoder = nn.ConvTranspose2d(in_channels=dim_inter, out_channels=(dim_in + dim_inter) // 2, kernel_size=(
            3, 3), stride=(1, 1), padding=(0, 0))
        self.decoder1 = nn.ConvTranspose2d(in_channels=(dim_in + dim_inter) // 2, out_channels=dim_in, kernel_size=(
            5, 5), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        x = self.encoder(x)
        # x = self.avgpool(x)
        x = self.bn1(x)
        x = self.decoder(x)
        x = self.decoder1(x)
        return x


if __name__ == "__main__":
    device = torch.device('cuda')
    model = TFblock_v3()
    model = model.to(device)
    input1 = torch.randn(8, 512, 4, 2)
    input1 = input1.to(device)
    outputs = model(input1)
    print(outputs.size())

    # device = torch.device('cuda')
    # model = PhaseNet()
    # model = model.to(device)
    #
    # input1 = torch.randn(8, 12, 10, 100)
    # input1 = input1.to(device)
    # # output = model(input1)
    # output = model(input1)
    # print(output.view((output.size(0), -1)).size())

    # device = torch.device('cuda')
    # # model = TimeDistributed(ImageNet())
    # # model = ImageDualNet()
    # # model = ImageFull(num_classes=7)
    # model = ImagePhaseNet(num_classes=7)
    # model = model.to(device)
    #
    # input1 = torch.randn(8, 100, 1, 91, 10)
    # input1 = input1.to(device)
    #
    # input2 = torch.randn(8, 100, 1, 91, 10)
    # input2 = input2.to(device)
    #
    # input3 = torch.randn(8, 12, 10, 100)
    # input3 = input3.to(device)
    # # output = model(input1)
    # # azi, ele = model(input1, input2)
    # out = model(input1, input2, input3)
    # print(out.size())

    # device = torch.device('cuda')
    # model = ImageNet()
    # model = ImageSingle_v1(num_classes=7, block=ImageDualNet_Single_v1, subblock=ImageNet_Large_v1)
    # model = ImageDualNet()
    # model = ImageFull(num_classes=7)
    # model = ImagePhaseNet_Single(num_classes=7)
    # model = ImagePhaseLSTM_Single(input_size=12, hidden_size=256, batch_size=8, num_classes=7)
    # model = ImageNet_Attention()

    # model = ImageDualNet_Attention_Single()
    # model = ImageFull(num_classes=7, block=ImageDualNet_Attention_Single, subblock=ImageNet_Attention_v1)
    # model = ImageNet_Square()
    # model = ImageFull_Square(num_classes=7, block=ImageDualNet_Single, subblock=ResNet10)
    # model = PhaseNet()

    # model = model.to(device)
    # summary(model, (1, 91, 10))
    # summary(model, (12, 10, 100))

    # input1 = torch.randn(8, 1, 91, 10)
    # input1 = torch.randn(8, 1, 70, 70)
    # input1 = input1.to(device)

    # input2 = torch.randn(8, 1, 91, 10)
    # input2 = torch.randn(8, 1, 70, 70)
    # input2 = input2.to(device)

    # input3 = torch.randn(8, 12, 10, 100)
    # input3 = torch.permute(input3, (0, 2, 3, 1))
    # input3 = torch.reshape(input3, (input3.size(0), -1, input3.size(3)))
    # input3 = input3.to(device)
    # out = model(input1)
    # azi, ele = model(input1, input2)
    # out, g = model(input1, input2)
    # out = model(input1, input2, input3)
    # print(out.size())
