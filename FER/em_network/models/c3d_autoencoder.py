import math
import torch
import torch.nn as nn
# from utils import device
from torchsummary import summary
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial


def Cos_similarity(x, y, dim=1):
    assert (x.shape == y.shape)

    if len(x.shape) >= 2:
        return F.cosine_similarity(x, y, dim=dim)
    else:
        return F.cosine_similarity(x.view(1, -1), y.view(1, -1))


def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


class SubNet(nn.Module):
    def __init__(self, act_mode='LeakyReLU'):
        super(SubNet, self).__init__()

        if act_mode == "ReLU":
            self.act_fun = nn.ReLU()
        elif act_mode == "LeakyReLU":
            self.act_fun = nn.LeakyReLU(1e-2)
        else:
            raise Exception("Not supported yet!")

        self.group1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(16),
            self.act_fun,
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 1)))
        self.group2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(32),
            self.act_fun,
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 1)))
        self.group3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            self.act_fun,
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            self.act_fun,
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 1, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            self.act_fun,
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            self.act_fun,
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 0)))

    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        return out


class SubNet_v1(nn.Module):
    def __init__(self, act_mode='LeakyReLU'):
        super(SubNet_v1, self).__init__()

        if act_mode == "ReLU":
            self.act_fun = nn.ReLU()
        elif act_mode == "LeakyReLU":
            self.act_fun = nn.LeakyReLU(1e-2)
        else:
            raise Exception("Not supported yet!")

        self.group1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(1e-2),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 1)))
        self.group2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(1e-2),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 1)))
        self.group3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(1e-2),
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(1e-2),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 1, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(1e-2),
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(1e-2),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 0)))

    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        return out



class Autoencoder_v2(nn.Module):
    def __init__(self, sample_duration):
        super(Autoencoder_v2, self).__init__()

        # encoder part
        self.net_azimuth = SubNet_v1()
        self.net_elevation = SubNet_v1()

        encoder_duration = int(math.floor(sample_duration / 8))
        encoder_channels = 64
        encoder_size_h = 2
        encoder_size_w = 2

        self.embedding = nn.Linear((encoder_channels * 2 * encoder_duration * encoder_size_h * encoder_size_w), 128)

        # self.encoder_fc1 = nn.Linear((encoder_channels * 2 * encoder_duration * encoder_size_h * encoder_size_w), 512)
        # self.encoder_fc2 = nn.Linear(512, 128)
        # self.encoder_fc3 = nn.Linear(128, 32)
        # # decoder part
        # self.decoder_fc1 = nn.Linear(32, 64)
        #
        # self.decoder1 = nn.Sequential(
        #     nn.ConvTranspose3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
        #     nn.ReLU(),
        # )
        #
        # self.decoder2 = nn.Sequential(
        #     nn.ConvTranspose3d(96, 128, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
        #     nn.ReLU(),
        # )
        #
        # self.decoder3 = nn.Sequential(
        #     nn.ConvTranspose3d(128, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        #     nn.ReLU(),
        # )

    def forward(self, azi, ele):
        # encoder
        out_azi = self.net_azimuth(azi)
        out_ele = self.net_elevation(ele)
        out_azi = out_azi.view(out_azi.size(0), -1)
        out_ele = out_ele.view(out_ele.size(0), -1)
        out = torch.cat((out_azi, out_ele), dim=1)
        out_embedding = self.embedding(out)

        return out_embedding


class Autoencoder(nn.Module):
    def __init__(self, sample_duration):
        super(Autoencoder, self).__init__()

        # encoder part
        self.net_azimuth = SubNet()
        self.net_elevation = SubNet()

        encoder_duration = int(math.floor(sample_duration / 8))
        encoder_channels = 64
        encoder_size_h = 2
        encoder_size_w = 2

        self.encoder_fc1 = nn.Linear((encoder_channels * 2 * encoder_duration * encoder_size_h * encoder_size_w), 512)
        self.encoder_fc2 = nn.Linear(512, 128)
        self.encoder_fc3 = nn.Linear(128, 32)
        # decoder part
        self.decoder_fc1 = nn.Linear(32, 64)

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.ReLU(),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(96, 128, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
        )

    def forward(self, azi, ele):
        # encoder
        out_azi = self.net_azimuth(azi)
        out_ele = self.net_elevation(ele)
        out = torch.cat((out_azi, out_ele), dim=1)
        out = out.view(out.size(0), -1)
        out = self.encoder_fc1(out)
        out = self.encoder_fc2(out)
        out_encoder = self.encoder_fc3(out)

        # decoder
        out = self.decoder_fc1(out_encoder)
        # out = self.decoder_fc2(out)
        out = out.view(out.size(0), -1, 1, 1, 1)
        out = self.decoder1(out)
        out = self.decoder2(out)
        out_decoder = self.decoder3(out)

        return out_encoder, out_decoder


class Classifier(nn.Module):
    def __init__(self, num_classes=7, decoder_duration=12,
                 decoder_channels=128, decoder_size=5):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear((decoder_channels * decoder_duration * decoder_size * decoder_size), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.classifier(input)
        return out


class ClassifierSmall(nn.Module):
    def __init__(self, num_classes=7):
        super(ClassifierSmall, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.classifier(input)
        return out


class AutoencoderWithClassifier(nn.Module):
    def __init__(self, sample_duration=100, num_classes=7):
        super(AutoencoderWithClassifier, self).__init__()
        decoder_duration = 2
        decoder_channels = 128
        decoder_size = 5
        self.autoencoder = Autoencoder(sample_duration)
        self.classifier = Classifier(num_classes, decoder_duration,
                                     decoder_channels, decoder_size)

    def forward(self, azi, ele):
        out_encoder, out_decoder = self.autoencoder(azi, ele)
        out = out_decoder.view(out_decoder.size(0), -1)
        out = self.classifier(out)
        return out_decoder, out


class AutoencoderWithClassifier_v2(nn.Module):
    def __init__(self, sample_duration=100, num_classes=7):
        super(AutoencoderWithClassifier_v2, self).__init__()
        self.autoencoder = Autoencoder_v2(sample_duration)
        self.classifier = ClassifierSmall(num_classes)

    def forward(self, azi, ele):
        out_embedding = self.autoencoder(azi, ele)
        out = self.classifier(out_embedding)
        return out_embedding, out


class RankingLossFunc(nn.Module):
    def __init__(self, delta):
        super(RankingLossFunc, self).__init__()
        self.delta = delta

    def forward(self, X, Y):
        assert (X.shape[0] == Y.shape[0])
        loss = 0
        num_of_samples = X.shape[0]

        mask = torch.eye(num_of_samples)
        for idx in range(0, num_of_samples):
            negative_sample_ids = [j for j in range(0, num_of_samples) if mask[idx][j] < 1]

            loss += sum([torch.sum(torch.max((self.delta
                                              - Cos_similarity(X[idx], Y[idx])
                                              + Cos_similarity(X[idx], Y[j])), 0).values) for j in negative_sample_ids])
        return loss


class Embedding_LOSS(nn.Module):
    def __init__(self):
        super(Embedding_LOSS, self).__init__()
        self.act_tanh = nn.Tanh()

    def forward(self, x):
        x = self.act_tanh(x)
        # x = norm(x)

        return x


if __name__ == "__main__":
    device = torch.device('cuda')
    # device = torch.device('cpu')
    model = AutoencoderWithClassifier()
    model = SubNet_v1()
    model = model.to(device)


    var_azi = torch.randn((8, 1, 100, 91, 10)).to(device)
    var_ele = torch.randn((8, 1, 100, 91, 10)).to(device)

    summary(model, (1, 100, 91, 10))

    # out = model(var_azi, var_ele)
    # print(out[0].size())
    # print(out[1].size())


    # model = Autoencoder_v2(sample_duration=100)
    # model = model.to(device)

    # summary(model,(8, 3, 30, 224, 224) )

    # var_azi = torch.randn((8, 1, 100, 91, 10)).to(device)
    # var_ele = torch.randn((8, 1, 100, 91, 10)).to(device)

    # output = model(var_azi, var_ele)
    # print(output.size())
