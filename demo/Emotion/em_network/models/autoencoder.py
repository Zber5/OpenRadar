from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        in_channel = 12
        out_channel = 16
        kernel_size = 3
        padding = [1, 1]

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[0])),
            # nn.BatchNorm2d(out1_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[1])),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[1])),

            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[0])),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[0])),
            nn.Upsample(scale_factor=(1, 2), mode="bilinear"),

            nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[0])),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(True),

            nn.Conv2d(out_channel, 11, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[0])),
            nn.AvgPool2d((1, 5)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        output = self.decoder(x)
        return output


class EMOEncode(nn.Module):
    def __init__(self):
        super(EMOEncode, self).__init__()
        self.conv1a = nn.Conv2d(in_channels=12, out_channels=12,
                                kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.conv1b = nn.Conv2d(in_channels=12, out_channels=12,
                                kernel_size=(2, 5), stride=(2, 2), padding=(1, 2))
        self.conv2a = nn.Conv2d(in_channels=12, out_channels=24,
                                kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.conv2b = nn.Conv2d(in_channels=24, out_channels=24,
                                kernel_size=(2, 5), stride=(2, 2), padding=(0, 2))
        self.conv3a = nn.Conv2d(in_channels=24, out_channels=48,
                                kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.conv3b = nn.Conv2d(in_channels=48, out_channels=48,
                                kernel_size=(1, 5), stride=(1, 2), padding=(0, 2))

        # self.skipconv1a = nn.Conv2d(in_channels=12, out_channels=12,
        #                             kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        # self.skipconv1b = nn.Conv2d(in_channels=12, out_channels=12,
        #                             kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        # self.skipconv2a = nn.Conv2d(in_channels=12, out_channels=24,
        #                             kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        # self.skipconv2b = nn.Conv2d(in_channels=24, out_channels=24,
        #                             kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        # self.skipconv3a = nn.Conv2d(in_channels=24, out_channels=48,
        #                             kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        # self.skipconv3b = nn.Conv2d(in_channels=48, out_channels=48,
        #                             kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))

        self.bn1a = nn.BatchNorm2d(num_features=12)
        self.bn1b = nn.BatchNorm2d(num_features=12)
        self.bn2a = nn.BatchNorm2d(num_features=24)
        self.bn2b = nn.BatchNorm2d(num_features=24)
        self.bn3a = nn.BatchNorm2d(num_features=48)
        self.bn3b = nn.BatchNorm2d(num_features=48)

        # self.skipbn1a = nn.BatchNorm2d(num_features=12)
        # self.skipbn1b = nn.BatchNorm2d(num_features=12)
        # self.skipbn2a = nn.BatchNorm2d(num_features=24)
        # self.skipbn2b = nn.BatchNorm2d(num_features=24)
        # self.skipbn3a = nn.BatchNorm2d(num_features=48)
        # self.skipbn3b = nn.BatchNorm2d(num_features=48)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 12, 3, 30) -> (B, 12, 3, 30)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 12, 3, 30) -> (B, 12, 2, 15)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 12, 2, 15) -> (B, 24, 2, 15)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 24, 1, 15) -> (B, 24, 1, 8)
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 24, 1, 8) -> (B, 48, 1, 8)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 48, 1, 8) -> (B, 48, 1, 4)
        x = x.view(x.size(0), -1)
        return x


class EMODecode(nn.Module):

    def __init__(self, n_class):
        super(EMODecode, self).__init__()
        self.convt1 = nn.ConvTranspose2d(in_channels=48, out_channels=32,
                                         kernel_size=(1, 6), stride=(1, 2), padding=(0, 2))
        self.convt2 = nn.ConvTranspose2d(in_channels=32, out_channels=24,
                                         kernel_size=(1, 6), stride=(1, 2), padding=(0, 2))
        self.convt3 = nn.ConvTranspose2d(in_channels=24, out_channels=n_class,
                                         kernel_size=(1, 6), stride=(1, 2), padding=(0, 2))
        # self.maxpool = nn.MaxPool2d()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(n_class*32, n_class*16)
        self.fc2 = nn.Linear(n_class*16, n_class*8)
        self.fc3 = nn.Linear(n_class*8, n_class)
        # self.upsample = nn.Upsample(size=(rodnet_configs['win_size'], radar_configs['ramap_rsize'],
        #                                   radar_configs['ramap_asize']), mode='nearest')

    # def forward(self, x, x1, x2, x3):
    #     x = self.prelu(self.convt1(x + x3))  # (B, 256, W/4, 16, 16) -> (B, 128, W/2, 32, 32)
    #     x = self.prelu(self.convt2(x + x2))  # (B, 128, W/2, 32, 32) -> (B, 64, W, 64, 64)
    #     x = self.convt3(x + x1)  # (B, 64, W, 64, 64) -> (B, 3, W, 128, 128)
    #     return x

    def forward(self, x):
        x = x.view(x.size(0), 48, 1, 4)
        x = self.prelu(self.convt1(x))  # (B, 48, 1, 4) -> (B, 32, 1, 8)
        x = self.prelu(self.convt2(x))  # (B, 32, 1, 8) -> (B, 24, 1, 16)
        x = self.convt3(x)  # (B, 24, 1, 16) -> (B, 20, 1, 32)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    x_train = torch.rand((32, 12, 3, 30))
    x_train = x_train.to(device)
    y_train = torch.rand((32, 1, 20))

    encoder = EMOEncode()
    decoder = EMODecode(20)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    x = encoder(x_train)
    x = decoder(x)

    print("")
