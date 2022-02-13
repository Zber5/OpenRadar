import numpy as np
import random
import os, errno
import sys
from tqdm import trange
from torchsummary import summary

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from FER.em_network.utils import device


class PhaseLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size, num_layers=2):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(PhaseLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # self.hidden = self.init_hidden()

    def forward(self, x_input):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''
        lstm_out, _ = self.lstm(x_input)
        out = lstm_out[:, -1, :]

        return out

    def init_hidden(self):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))


class PhaseLSTM_v1(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size, num_layers=2):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(PhaseLSTM_v1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # self.hidden = self.init_hidden()

    def forward(self, x_input):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''
        lstm_out, _ = self.lstm(x_input)
        out = lstm_out[:, -1, :]

        return out

    def init_hidden(self):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))


if __name__ == "__main__":
    # model = TimeDistributed(ImageNet())
    # model = ImageDualNet()
    # model = ImageFull(num_classes=7)
    # model = ImagePhaseNet_Single(num_classes=7)
    model = PhaseLSTM(input_size=468 * 2, hidden_size=468 * 2, batch_size=2)
    # model = model.to(device)

    # input1 = torch.randn(2, 30, 468 * 2)
    # input1 = input1.to(device)
    #
    # input2 = torch.randn(8, 1, 91, 10)
    # input2 = input2.to(device)
    # input3 = torch.randn(8, 12, 10, 100)
    # input3 = input3.to(device)
    # output = model(input1)
    # azi, ele = model(input1, input2)
    # out = model(input1)
    # print(out.size())
    # print(out.size())
