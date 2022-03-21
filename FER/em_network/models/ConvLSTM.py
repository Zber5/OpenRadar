"""
source code from:
https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

"""

import torch.nn as nn
import torch
from torchsummary import summary
import torch.nn.functional as F
from torch.nn import init


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        # self.padding = (0, 0)
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.maxpool1 = nn.MaxPool3d((3, 2, 2), stride=(1, 2, 2), padding=())

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTMFull(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, num_classes,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTMFull, self).__init__()
        self.feature_azi = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                                    batch_first, bias, return_all_layers)
        self.feature_ele = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                                    batch_first, bias, return_all_layers)
        self.fc1 = nn.Sequential(
            nn.Linear((8 * 91 * 10 * 2), 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, num_classes))

    def forward(self, azi, ele):
        # azi
        _, azi_last = self.feature_azi(azi)
        _, azi_x = azi_last[0]
        azi_x = azi_x.view(azi_x.size(0), -1)
        # ele
        _, ele_last = self.feature_ele(ele)
        _, ele_x = ele_last[0]
        ele_x = ele_x.view(ele_x.size(0), -1)

        x = torch.cat((azi_x, ele_x), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class ConvLSTMFull_v1(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, num_classes,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTMFull_v1, self).__init__()
        self.feature_azi = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                                    batch_first, bias, return_all_layers)
        self.feature_ele = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                                    batch_first, bias, return_all_layers)
        self.fc1 = nn.Sequential(
            nn.Linear((8 * 91 * 10 * 2), 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, num_classes))

    def forward(self, azi, ele):
        # azi
        azi_out, _ = self.feature_azi(azi)
        azi_x = azi_out[0][:, -1, :, :, :]
        azi_x = azi_x.view(azi_x.size(0), -1)
        # ele
        ele_out, _ = self.feature_ele(ele)
        ele_x = ele_out[0][:, -1, :, :, :]
        ele_x = ele_x.view(ele_x.size(0), -1)

        x = torch.cat((azi_x, ele_x), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class TemporalAttention(nn.Module):

    def __init__(self, channel=100, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, t, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, t)
        y = self.fc(y).view(b, t, 1, 1, 1)
        return x * y.expand_as(x)


class ConvLSTMFull_ME(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTMFull_ME, self).__init__()
        self.feature_azi = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                                    batch_first, bias, return_all_layers)
        self.feature_ele = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                                    batch_first, bias, return_all_layers)
        self.temporal = TemporalAttention(channel=100, reduction=8)

        self.r_azi = torch.nn.Parameter(torch.ones((1, 100, 1, 1, 1)))
        self.r_ele = torch.nn.Parameter(torch.ones((1, 100, 1, 1, 1)))

        self.eps = 1e-5

    def forward(self, azi, ele):
        # azi
        azi_out, _ = self.feature_azi(azi)
        # azi_x = self.temporal(azi_out[0])
        # azi_x = (azi_out[0] - torch.mean(azi_out[0], dim=(2, 3, 4), keepdim=True)) / torch.sqrt(
        #     torch.var(azi_out[0], dim=(2, 3, 4), unbiased=False, keepdim=True) + self.eps) * self.r_azi

        # azi_x = (azi_out[0] - torch.mean(azi_out[0], dim=1, keepdim=True)) / torch.sqrt(
        #     torch.var(azi_out[0], dim=1, unbiased=False, keepdim=True) + self.eps) * self.r_azi

        azi_x = torch.sum(azi_out[0], dim=1)
        # azi_x1, _ = torch.max(azi_out[0], dim=1)
        # azi_x2 = torch.mean(azi_out[0], dim=1)

        # azi_x = torch.sum(azi_x, dim=1)
        # azi_x1, _ = torch.max(azi_x, dim=1)
        # azi_x2 = torch.mean(azi_x, dim=1)
        # azi_x = azi_x1 + azi_x2
        azi_x = F.normalize(azi_x)

        # ele
        ele_out, _ = self.feature_ele(ele)
        # ele_x = self.temporal(ele_out[0])
        # ele_x = (ele_out[0] - torch.mean(ele_out[0], dim=(2, 3, 4), keepdim=True)) / torch.sqrt(
        #     torch.var(ele_out[0], dim=(2, 3, 4), keepdim=True) + self.eps) * self.r_ele
        # ele_x = (ele_out[0] - torch.mean(ele_out[0], dim=1, keepdim=True)) / torch.sqrt(
        #     torch.var(ele_out[0], dim=1, keepdim=True) + self.eps) * self.r_ele
        # ele_x = torch.sum(ele_out[0], dim=1)
        # ele_x1, _ = torch.max(ele_out[0], dim=1)
        # ele_x2 = torch.mean(ele_out[0], dim=1)

        ele_x = torch.sum(ele_out[0], dim=1)
        # ele_x = torch.sum(ele_x, dim=1)
        # ele_x1, _ = torch.max(ele_x, dim=1)
        # ele_x2 = torch.mean(ele_x, dim=1)
        # ele_x = ele_x1 + ele_x2
        ele_x = F.normalize(ele_x)

        return azi_x, ele_x


if __name__ == "__main__":
    device = torch.device('cuda')
    channels = 1

    input1 = torch.rand((8, 100, 1, 91, 10))
    input1 = input1.to(device)
    input2 = torch.rand((8, 100, 1, 91, 10))
    input2 = input2.to(device)

    # model = ConvLSTM(input_dim=channels,
    #                  hidden_dim=[2, 4, 8],
    #                  kernel_size=(7, 3),
    #                  num_layers=3,
    #                  batch_first=True,
    #                  bias=True,
    #                  return_all_layers=False)
    #
    # model = model.to(device)
    # print(model)
    # layer_output_list, last_state_list = model(input)
    #
    # for layer in layer_output_list:
    #     print(layer.size())
    #
    # print("\n\n\n")
    # for last in last_state_list:
    #     print(last[0].size())
    #     print(last[1].size())

    model = ConvLSTMFull_ME(input_dim=channels,
                            hidden_dim=[3],
                            kernel_size=(1, 1),
                            num_layers=1,
                            batch_first=True,
                            bias=True,
                            return_all_layers=False)

    model = model.to(device)

    out1, out2 = model(input1, input2)
    print(out1.size())
    print(out2.size())
