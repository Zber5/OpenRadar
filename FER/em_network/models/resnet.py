import torch
import torch.nn as nn
import torch.nn.functional as F
from FER.em_network.utils import model_parameters
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from FER.em_network.models.model import TimeDistributed


# class ResNetX(ResNet):
#     def __init__(self, block, layers, num_classes=7, **kwargs):
#         super(ResNetX, self).__init__(block, layers, num_classes, **kwargs)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)  # 64,56,56
#
#         x = self.layer1(x)  # 64, 56, 56
#         x = self.layer2(x)  # 128, 28, 28
#         x = self.layer3(x)  # 256, 14, 14
#         x = self.layer4(x)  # 512, 7, 7
#
#         x = self.avgpool(x)
#         return x


class ResNetX(ResNet):
    def __init__(self, block, layers, num_classes=7, **kwargs):
        super(ResNetX, self).__init__(block, layers, num_classes, **kwargs)
        self.t_conv1 = TimeDistributed(self.conv1)
        self.t_bn1 = TimeDistributed(self.bn1)
        self.t_relu = TimeDistributed(self.relu)
        self.t_maxpool = TimeDistributed(self.maxpool)

        self.t_layer1 = TimeDistributed(self.layer1)
        self.t_layer2 = TimeDistributed(self.layer2)
        self.t_layer3 = TimeDistributed(self.layer3)
        self.t_layer4 = TimeDistributed(self.layer4)
        self.t_avgpool = TimeDistributed(self.avgpool)

    def forward(self, x):
        x = self.t_conv1(x)
        x = self.t_bn1(x)
        x = self.t_relu(x)
        x = self.t_maxpool(x)  # 64,56,56

        g1 = self.t_layer1(x)  # 64, 56, 56
        g2 = self.t_layer2(g1)  # 128, 28, 28
        g3 = self.t_layer3(g2)  # 256, 14, 14
        g4 = self.t_layer4(g3)  # 512, 7, 7

        x = self.t_avgpool(g4)
        # x = g4
        return x, (g1, g2, g3, g4)


def _resnet(
        arch: str,
        block,
        layers,
        pretrained: bool,
        progress: bool,
        **kwargs):
    model = ResNetX(block, layers, **kwargs)

    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet10(pretrained: bool = False, progress: bool = True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet10', BasicBlock, [2, 1, 2, 1], pretrained, progress,
                   **kwargs)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        print(x.shape)
        print(identity.shape)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_channels=1):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=3, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


class Classifier(nn.Module):
    def __init__(self, num_classes=7):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, input):
        out = self.classifier(input)
        return out


class ResNetFull(nn.Module):
    def __init__(self, fmodel, cmodel):
        super(ResNetFull, self).__init__()
        self.feature = TimeDistributed(fmodel)
        self.classifier = cmodel

    def forward(self, x):
        out = self.feature(x)
        out = torch.mean(out, dim=1)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class ResNetFull_Teacher(nn.Module):
    def __init__(self, fmodel, cmodel):
        super(ResNetFull_Teacher, self).__init__()
        self.feature = fmodel
        self.classifier = cmodel

    def forward(self, x):
        out, gs = self.feature(x)
        out = torch.mean(out, dim=1)
        out = torch.flatten(out, 1)
        out1 = self.classifier(out)

        return out1, gs


def ResNet18(channels=3):
    return ResNet(Bottleneck, [2, 2, 2, 2], channels)


def ResNet10():
    return ResNet(Bottleneck, [2, 2, 2, 2])


def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)


def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)


def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels)


if __name__ == "__main__":
    device = torch.device('cuda')

    _structure = resnet18()
    _parameterDir = "C:/Users/Zber/Documents/GitHub/Emotion-FAN/pretrain_model/Resnet18_FER+_pytorch.pth.tar"
    fmodel = model_parameters(_structure, _parameterDir)
    cmodel = Classifier(num_classes=7)
    model = ResNetFull_Teacher(fmodel, cmodel)
    model = model.to(device)
    input1 = torch.randn(8, 30, 3, 224, 224)
    # input1 = torch.randn(8, 3, 224, 224)
    input1 = input1.to(device)

    out, g = model(input1)

    print(out.size())

    # device = torch.device('cuda')
    # fmodel = resnet10()
    # # _parameterDir = "C:/Users/Zber/Documents/GitHub/Emotion-FAN/pretrain_model/Resnet18_FER+_pytorch.pth.tar"
    # # fmodel = model_parameters(_structure, _parameterDir)
    # cmodel = Classifier(num_classes=7)
    # model = ResNetFull(fmodel, cmodel)
    # model = model.to(device)
    # input1 = torch.randn(8, 1, 1, 70, 70)
    # # input1 = torch.randn(8, 3, 224, 224)
    # input1 = input1.to(device)
    #
    # out = model(input1)
    #
    # print(out.size())
