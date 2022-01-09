import torch
import torch.nn as nn
import torch.nn.functional as F

from FER.em_network.models.Conv2D import ImageNet, ImageFull, ImageDualNet
from FER.em_network.models.ConvLSTM import ConvLSTM


class TimeSpaceNet(nn.Module):
    def __init__(self, num_classes):
        super(TimeSpaceNet, self).__init__()
        self.lstm_azi = ConvLSTM(input_dim=1,
                                 hidden_dim=[2, 4, 8],
                                 kernel_size=(7, 3),
                                 num_layers=3,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)
        self.conv_azi = ImageNet(num_channel=8)

        self.lstm_ele = ConvLSTM(input_dim=1,
                                 hidden_dim=[2, 4, 8],
                                 kernel_size=(7, 3),
                                 num_layers=3,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)

        self.conv_ele = ImageNet(num_channel=8)
        self.img_net = ImageDualNet()

        self.fc1 = nn.Sequential(
            nn.Linear(16608, 2048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(32, num_classes),
            nn.Sigmoid())

    def forward(self, azi, ele):
        _, azi_last = self.lstm_azi(azi)
        _, azi_lstm_out = azi_last[0]

        _, ele_last = self.lstm_ele(ele)
        _, ele_lstm_out = ele_last[0]

        # azi_lstm_out = self.conv_azi(azi_lstm_out)
        azi_lstm_out = azi_lstm_out.view((azi_lstm_out.size(0), -1))
        ele_lstm_out = ele_lstm_out.view((ele_lstm_out.size(0), -1))

        lstm_out = torch.cat((azi_lstm_out, ele_lstm_out), dim=1)

        azi_out, ele_out = self.img_net(azi, ele)

        img_out = torch.cat((azi_out, ele_out), dim=1)
        out = torch.cat((img_out, lstm_out), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out


class TimeSpaceNet_v1(nn.Module):
    def __init__(self, num_classes):
        super(TimeSpaceNet_v1, self).__init__()
        self.lstm_azi = ConvLSTM(input_dim=1,
                                 hidden_dim=[2, 4, 8],
                                 kernel_size=(7, 3),
                                 num_layers=3,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)
        self.conv_azi = ImageNet(num_channel=8)
        self.img_net = ImageDualNet()

        self.fc1 = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(32, num_classes),
            nn.Sigmoid())

    def forward(self, azi, ele):
        _, azi_last = self.lstm_azi(azi)
        _, azi_lstm_out = azi_last[0]

        azi_lstm_out = self.conv_azi(azi_lstm_out)
        lstm_out = azi_lstm_out.view((azi_lstm_out.size(0), -1))

        azi_out, ele_out = self.img_net(azi, ele)

        img_out = torch.cat((azi_out, ele_out), dim=1)
        out = torch.cat((img_out, lstm_out), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


if __name__ == "__main__":
    device = torch.device('cuda')
    model = TimeSpaceNet(num_classes=7)
    model = model.to(device)

    input1 = torch.randn(8, 100, 1, 91, 10)
    input1 = input1.to(device)

    input2 = torch.randn(8, 100, 1, 91, 10)
    input2 = input2.to(device)

    out = model(input1, input2)
    print(out.size())
