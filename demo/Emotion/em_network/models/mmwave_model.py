from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from demo.Emotion.em_network.utils import device
from torch.nn import LSTM
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import time

# set seed, make result reporducable
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



class mmwave_lstm(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, batch_size, num_classes=7, num_layers=2):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(mmwave_lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.hidden = self.init_hidden()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_input):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''
        lstm_out, self.hidden = self.lstm(x_input, self.hidden)
        output = self.sigmoid(self.linear(lstm_out[-1]))


        return output, self.hidden

    def init_hidden(self):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size, hidden_size, batch_size, num_classes = 36, 64, 32, 7
    lstm = mmwave_lstm(input_size, hidden_size, batch_size, num_classes=num_classes)
    input = torch.rand((32, 36, 300))
    input = input.permute(2, 0, 1)

    output, hidden = lstm(input)

    print("")
