"""
This is the c3d implementation with batch norm.

References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks."
Proceedings of the IEEE international conference on computer vision. 2015.

"""

"""
1.C3D with one directional heatmap  -> C3D
2.C3D with two directional heatmaps  -> C3DFusionBaseline
3.C3D with two directional heatmaps fusion -> C3DFusionV2
4.C3D with multimodal phase attention -> ATT_PHASE
5.MMTM add to C3D multimodal fusion + phase attention
  https://github.com/haamoon/mmtm/blob/master/mmtm.py
"""

import math
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        n, t = x.size(0), x.size(1)
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(n * t, x.size(2), x.size(3), x.size(4))
        y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(n, t, y.size(1), y.size(2), y.size(3))
        return y


class TimeDistributedTwin(nn.Module):
    def __init__(self, module):
        super(TimeDistributedTwin, self).__init__()
        self.module = module

    def forward(self, x, z):
        if len(x.size()) <= 2:
            return self.module(x)
        xn, xt = x.size(0), x.size(1)
        zn, zt = z.size(0), z.size(1)
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(xn * xt, x.size(2), x.size(3), x.size(4))
        z_reshape = z.contiguous().view(zn * zt, z.size(2), z.size(3), z.size(4))
        y, v = self.module(x_reshape, z_reshape)
        # We have to reshape Y
        y = y.contiguous().view(xn, xt, y.size(1), y.size(2), y.size(3))
        v = v.contiguous().view(zn, zt, v.size(1), v.size(2), v.size(3))
        return y, v


def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        print(m.weight)
    else:
        print('error')


class MMTM(nn.Module):
    def __init__(self, dim_visual, dim_skeleton, ratio):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_skeleton
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # initialize
        # with torch.no_grad():
        #     self.fc_squeeze.apply(init_weights)
        #     self.fc_visual.apply(init_weights)
        #     self.fc_skeleton.apply(init_weights)

    def forward(self, visual, skeleton):
        squeeze_array = []
        for tensor in [visual, skeleton]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))
        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

        return visual * vis_out, skeleton * sk_out


class C3D(nn.Module):
    def __init__(self,
                 sample_duration,
                 num_classes=600):
        super(C3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 1)))
        self.group2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 1)))
        self.group3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 1, 2)))
        # self.group4 = nn.Sequential(
        #     nn.Conv3d(64, 256, kernel_size=(3, 7, 3), padding=1),
        #     nn.BatchNorm3d(256),
        #     nn.ReLU(),
        #     nn.Conv3d(256, 256, kernel_size=(3, 7, 3), padding=1),
        #     nn.BatchNorm3d(256),
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 0)))

        last_duration = int(math.floor(sample_duration / 8))
        # last_size = int(math.ceil(sample_size / 32))
        last_size_h = 2
        last_size_w = 2
        self.fc1 = nn.Sequential(
            nn.Linear((64 * last_duration * last_size_h * last_size_w), 1024),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(256, num_classes))

    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        # out = self.group5(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


class SubNet(nn.Module):
    def __init__(self):
        super(SubNet, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 1)))
        self.group2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 1)))
        self.group3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 1, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 0)))

    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        return out


class C3DFusionBaseline(nn.Module):
    def __init__(self,
                 sample_duration,
                 num_classes=600):
        super(C3DFusionBaseline, self).__init__()
        self.net_azimuth = SubNet()
        self.net_elevation = SubNet()

        last_duration = int(math.floor(sample_duration / 8))
        last_size_h = 2
        last_size_w = 2
        self.fc1 = nn.Sequential(
            nn.Linear((128 * last_duration * last_size_h * last_size_w), 1024),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5))
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


class C3DFusionBaseline_out(nn.Module):
    def __init__(self,
                 sample_duration,
                 num_classes=600):
        super(C3DFusionBaseline_out, self).__init__()
        self.net_azimuth = SubNet()
        self.net_elevation = SubNet()

        last_duration = int(math.floor(sample_duration / 8))
        last_size_h = 2
        last_size_w = 2
        self.fc1 = nn.Sequential(
            nn.Linear((128 * last_duration * last_size_h * last_size_w), 1024),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(256, num_classes))

    def forward(self, azi, ele):
        out_azi = self.net_azimuth(azi)
        out_ele = self.net_elevation(ele)

        # concatenation
        out1 = torch.cat((out_azi, out_ele), dim=1)

        out = out1.view(out1.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out1, out


class C3DMMTM_v1(nn.Module):
    def __init__(self,
                 sample_duration,
                 num_classes=600):
        super(C3DMMTM_v1, self).__init__()
        self.net_azimuth = SubNet()
        self.net_elevation = SubNet()
        self.mmtm = TimeDistributedTwin(MMTM(64, 64, 4))

        last_duration = int(math.floor(sample_duration / 8))
        last_size_h = 2
        last_size_w = 2

        self.fc1 = nn.Sequential(
            nn.Linear((128 * last_duration * last_size_h * last_size_w), 1024),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(256, num_classes))

    def forward(self, azi, ele):
        out_azi = self.net_azimuth(azi)
        out_ele = self.net_elevation(ele)

        # MMTM fusion
        out_azi = out_azi.permute(0, 2, 1, 3, 4)
        out_ele = out_ele.permute(0, 2, 1, 3, 4)

        out_azi, out_ele = self.mmtm(out_azi, out_ele)
        out_azi = out_azi.permute(0, 2, 1, 3, 4)
        out_ele = out_ele.permute(0, 2, 1, 3, 4)

        # concatenation
        out = torch.cat((out_azi, out_ele), dim=1)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)


        return out



class C3DMMTM_v2(nn.Module):
    def __init__(self,
                 sample_duration,
                 num_classes=600):
        super(C3DMMTM_v2, self).__init__()

        self.azi_group1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 1)))
        self.azi_group2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 1)))
        self.azi_group3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 1, 2)))
        self.azi_group4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 0)))

        self.ele_group1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 1)))
        self.ele_group2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 1)))
        self.ele_group3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 1, 2)))
        self.ele_group4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 7, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 0)))

        self.mmtm1 = TimeDistributedTwin(MMTM(16, 16, 4))
        self.mmtm2 = TimeDistributedTwin(MMTM(32, 32, 4))
        self.mmtm3 = TimeDistributedTwin(MMTM(64, 64, 4))
        self.mmtm4 = TimeDistributedTwin(MMTM(64, 64, 4))

        last_duration = int(math.floor(sample_duration / 8))
        last_size_h = 2
        last_size_w = 2

        self.fc1 = nn.Sequential(
            nn.Linear((128 * last_duration * last_size_h * last_size_w), 1024),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(256, num_classes))


    def forward(self, azi, ele):

        # group 1
        out_azi = self.azi_group1(azi)
        out_ele = self.ele_group1(ele)

        # MMTM fusion1
        out_azi = out_azi.permute(0, 2, 1, 3, 4)
        out_ele = out_ele.permute(0, 2, 1, 3, 4)

        out_azi, out_ele = self.mmtm1(out_azi, out_ele)
        out_azi = out_azi.permute(0, 2, 1, 3, 4)
        out_ele = out_ele.permute(0, 2, 1, 3, 4)

        # group 2
        out_azi = self.azi_group2(out_azi)
        out_ele = self.ele_group2(out_ele)

        # MMTM fusion2
        out_azi = out_azi.permute(0, 2, 1, 3, 4)
        out_ele = out_ele.permute(0, 2, 1, 3, 4)

        out_azi, out_ele = self.mmtm2(out_azi, out_ele)
        out_azi = out_azi.permute(0, 2, 1, 3, 4)
        out_ele = out_ele.permute(0, 2, 1, 3, 4)

        # group 3
        out_azi = self.azi_group3(out_azi)
        out_ele = self.ele_group3(out_ele)

        # MMTM fusion3
        out_azi = out_azi.permute(0, 2, 1, 3, 4)
        out_ele = out_ele.permute(0, 2, 1, 3, 4)

        out_azi, out_ele = self.mmtm3(out_azi, out_ele)
        out_azi = out_azi.permute(0, 2, 1, 3, 4)
        out_ele = out_ele.permute(0, 2, 1, 3, 4)

        # group 4
        out_azi = self.azi_group4(out_azi)
        out_ele = self.ele_group4(out_ele)

        # MMTM fusion3
        out_azi = out_azi.permute(0, 2, 1, 3, 4)
        out_ele = out_ele.permute(0, 2, 1, 3, 4)

        out_azi, out_ele = self.mmtm4(out_azi, out_ele)
        out_azi = out_azi.permute(0, 2, 1, 3, 4)
        out_ele = out_ele.permute(0, 2, 1, 3, 4)

        # concatenation
        out = torch.cat((out_azi, out_ele), dim=1)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)

        return out



# still the old version
class C3DAttention(nn.Module):
    def __init__(self,
                 sample_duration,
                 num_classes=600):
        super(C3DAttention, self).__init__()
        self.net_azimuth = SubNet()
        self.net_elevation = SubNet()

        last_duration = int(math.floor(sample_duration / 8))
        last_size_h = 2
        last_size_w = 2
        self.fc1 = nn.Sequential(
            nn.Linear((128 * last_duration * last_size_h * last_size_w), 1024),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5))
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


class C3DFusionV2(nn.Module):
    def __init__(self,
                 sample_duration,
                 num_classes=600):
        super(C3DFusionV2, self).__init__()
        self.net_azimuth = SubNet()
        self.net_elevation = SubNet()

        last_duration = int(math.floor(sample_duration / 8))
        last_size_h = 2
        last_size_w = 2

        self.fc_azi = nn.Sequential(
            nn.Linear((64 * last_duration * last_size_h * last_size_w), 1024),
            nn.ReLU(),
            nn.Sigmoid(),
        )
        self.fc_ele = nn.Sequential(
            nn.Linear((64 * last_duration * last_size_h * last_size_w), 1024),
            nn.ReLU(),
            nn.Sigmoid())

        self.fc1 = nn.Sequential(
            nn.Linear(1024 * 2, 1024),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(256, num_classes))

    def forward(self, azi, ele):
        out_azi = self.net_azimuth(azi)
        out_ele = self.net_elevation(ele)

        # azi
        out_azi = out_azi.view(out_azi.size(0), -1)
        out_azi = self.fc_azi(out_azi)

        # ele
        out_ele = out_ele.view(out_ele.size(0), -1)
        out_ele = self.fc_ele(out_ele)

        # concatenation
        out = torch.cat((out_azi, out_ele), dim=1)
        # out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


class ATT_PHASE(nn.Module):
    def __init__(self):
        super(ATT_PHASE, self).__init__()

        self.conv1 = TimeDistributed(nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(5, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU()))
        self.max1 = TimeDistributed(nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))

        self.conv2 = TimeDistributed(nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(3, 3), padding=(0, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU()))

        self.max2 = TimeDistributed(nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)))
        self.pool = nn.MaxPool3d(kernel_size=(8, 1, 1), stride=(1, 1, 1))

    def forward(self, x):
        out = self.conv1(x)  # (n, 4, 10, 5)
        out = self.max1(out)  # (n, 4, 5, 3)
        out = self.conv2(out)  # (n, 4, 3, 3)
        out = self.max2(out)
        out = self.pool(out)
        out = out.view(out.size(0), out.size(1))
        return out


class C3D_VIDEO(nn.Module):
    def __init__(self,
                 sample_size,
                 sample_duration,
                 num_classes=600):
        super(C3D_VIDEO, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(3, 5, 5), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 4, 4)))
        self.group4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 5, 5), padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=(3, 5, 5), padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 4, 4)))
        # self.group5 = nn.Sequential(
        #     nn.Conv3d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(512),
        #     nn.ReLU(),
        #     nn.Conv3d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(512),
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))

        last_duration = 1
        last_size = 1
        self.fc1 = nn.Sequential(
            nn.Linear((256 * last_duration * last_size * last_size), 128),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(32, num_classes))

    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        # out = self.group5(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


class C3D_VIDEO_out(nn.Module):
    def __init__(self,
                 sample_size,
                 sample_duration,
                 num_classes=600):
        super(C3D_VIDEO_out, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(3, 5, 5), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 4, 4)))
        self.group4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 5, 5), padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=(3, 5, 5), padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 4, 4)))

        last_duration = 1
        last_size = 1
        self.fc1 = nn.Sequential(
            nn.Linear((256 * last_duration * last_size * last_size), 128),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(32, num_classes))

    def forward(self, x):
        out1 = self.group1(x)
        out2 = self.group2(out1)
        out3 = self.group3(out2)
        out4 = self.group4(out3)
        # out = self.group5(out)
        out = out4.view(out4.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out4, out


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


if __name__ == '__main__':
    device = torch.device('cpu')

    model = C3DMMTM_v2(sample_duration=100, num_classes=7)
    model = model.to(device)

    var_azi = torch.randn((8, 1, 100, 91, 10)).to(device)
    var_ele = torch.randn((8, 1, 100, 91, 10)).to(device)
    output = model(var_azi, var_ele)
    print(output.shape)

    # model = C3DFusionV2(sample_duration=300, num_classes=6)
    # model = model.to(device)
    #
    # var_azi = torch.randn((8, 1, 300, 91, 10)).to(device)
    # var_ele = torch.randn((8, 1, 300, 91, 10)).to(device)
    # output = model(var_azi, var_ele)
    # print(output.shape)

    # torch summary
    # summary(model, ((1, 300, 91, 10), (1, 300, 91, 10)))

    # model = C3D_VIDEO(sample_size = 112, sample_duration = 16, num_classes=6)
    # model = model.to(device)
    #
    # input = torch.randn(8, 3, 16, 112, 112)
    # input = input.to(device)
    #
    # output = model(input)

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
