import math
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial


class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                                    squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                                    squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor


class SubNet_Attention(nn.Module):
    def __init__(self):
        super(SubNet_Attention, self).__init__()
        # self.att_fun = ChannelSpatialSELayer3D
        self.att_fun = ProjectExciteLayer
        self.group1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 1)))
        self.att1 = self.att_fun(16)
        self.group2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 1)))
        self.att2 = self.att_fun(32)
        self.group3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 1, 2)))
        self.att3 = self.att_fun(64)
        self.group4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 0)))
        # self.att4 = self.att_fun(64)

    def forward(self, x):
        out = self.group1(x)
        out = self.att1(out)
        out = self.group2(out)
        out = self.att2(out)
        out = self.group3(out)
        out = self.att3(out)
        out = self.group4(out)
        # out = self.att4(out)
        return out


class C3DFusionAttention(nn.Module):
    def __init__(self,
                 sample_duration,
                 num_classes=600):
        super(C3DFusionAttention, self).__init__()
        self.net_azimuth = SubNet_Attention()
        self.net_elevation = SubNet_Attention()

        last_duration = int(math.floor(sample_duration / 8))
        last_size_h = 2
        last_size_w = 2
        self.fc1 = nn.Sequential(
            nn.Linear((128 * last_duration * last_size_h * last_size_w), 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, num_classes))

    def forward(self, azi, ele):
        out_azi = self.net_azimuth(azi)
        out_ele = self.net_elevation(ele)

        # concatenation
        out = torch.cat((out_azi, out_ele), dim=1)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda')
    # device = torch.device('cpu')

    # model = C3DMMTM_v2(sample_duration=100, num_classes=7)
    # model = model.to(device)
    #
    # var_azi = torch.randn((8, 1, 100, 91, 10)).to(device)
    # var_ele = torch.randn((8, 1, 100, 91, 10)).to(device)
    # output = model(var_azi, var_ele)
    # print(output.shape)

    # model = C3DFusionV2(sample_duration=300, num_classes=6)
    # model = model.to(device)
    #
    # var_azi = torch.randn((8, 1, 300, 91, 10)).to(device)
    # var_ele = torch.randn((8, 1, 300, 91, 10)).to(device)
    # output = model(var_azi, var_ele)
    # print(output.shape)

    # torch summary
    # summary(model, ((1, 300, 91, 10), (1, 300, 91, 10)))

    # model = C3D_VIDEO_V3(sample_size=224, sample_duration=30, num_classes=7)
    # model = model.to(device)

    # summary(model,(8, 3, 30, 224, 224) )

    # input = torch.randn(8, 3, 30, 224, 224)
    # input = input.to(device)

    # output = model(input)
    # print(output.size())

    # model = ATT_PHASE()
    # model = model.to(device)
    #
    # input = torch.randn(8, 300, 1, 12, 5)
    # input = input.to(device)
    #
    # output = model(input)
    # print(output.size())

    # model = TimeDistributedTwin(MMTM(64, 64, 4))
    # model = model.to(device)
    #
    # input1 = torch.randn(8, 50, 8, 4, 4)
    # input2 = torch.randn(8, 50, 8, 2, 2)
    # input1 = torch.randn(8, 64, 12, 2, 2)
    # input2 = torch.randn(8, 64, 12, 2, 2)
    # input1 = input1.to(device)
    # input2 = input2.to(device)
    # input1 = input1.permute(0, 2, 1, 3, 4)
    # input2 = input2.permute(0, 2, 1, 3, 4)
    #
    # output1, output2 = model(input1, input2)
    # output1 = output1.permute(0, 2, 1, 3, 4)
    # output2 = output2.permute(0, 2, 1, 3, 4)
    # print(output1.size())
    # print(output2.size())

    # model = SubNet_v2()
    # model = model.to(device)
    #
    # input1 = torch.randn(8, 1, 100, 91, 50)
    # input1 = input1.to(device)
    # output = model(input1)
    # print(output.size())

    model = C3DFusionAttention(sample_duration=100, num_classes=7)
    model = model.to(device)

    input1 = torch.randn(8, 1, 100, 91, 10)
    input2 = torch.randn(8, 1, 100, 91, 10)
    input1 = input1.to(device)
    input2 = input2.to(device)

    output = model(input1, input2)
    print(output.size())
