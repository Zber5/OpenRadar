from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import time


class BRNN(nn.RNN):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1, bias=True, batch_first=True,
                 bidirectional=True):
        super(BRNN, self).__init__(input_size=input_size, hidden_size=hidden_size, bias=bias, batch_first=batch_first,
                                   bidirectional=bidirectional)
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional

    def init_hidden_size(self):
        if self.bidirectional:
            n = 2
        else:
            n = 1

        return torch.zeros(n * self.num_layers, self.batch_size, self.hidden_size)


class PHRNN(nn.Module):
    def __init__(self, eye_size, eyebrow_size, nose_size, mouth_size, h1_size, h2_size, h3_size, h4_size, h5_size,
                 h6_size, total_length=30, bidirectional=True, num_classes=6):
        super(PHRNN, self).__init__()

        self.length = total_length

        self.rnn1_1 = nn.RNN(input_size=eye_size, hidden_size=h1_size, bias=True, batch_first=True,
                             bidirectional=True)
        self.rnn1_2 = nn.RNN(input_size=eyebrow_size, hidden_size=h1_size, bias=True, batch_first=True,
                             bidirectional=True)
        self.rnn1_3 = nn.RNN(input_size=nose_size, hidden_size=h1_size, bias=True, batch_first=True,
                             bidirectional=True)
        self.rnn1_4 = nn.RNN(input_size=mouth_size, hidden_size=h1_size, bias=True, batch_first=True,
                             bidirectional=True)
        if bidirectional:
            h1_size = h1_size * 2

        self.rnn2_1 = nn.RNN(input_size=h1_size * 2, hidden_size=h2_size, bias=True, batch_first=True,
                             bidirectional=True)
        self.rnn2_2 = nn.RNN(input_size=h1_size, hidden_size=h2_size, bias=True, batch_first=True, bidirectional=True)
        self.rnn2_3 = nn.RNN(input_size=h1_size, hidden_size=h2_size, bias=True, batch_first=True, bidirectional=True)

        if bidirectional:
            h2_size = h2_size * 2

        self.rnn3_1 = nn.RNN(input_size=h2_size, hidden_size=h3_size, bias=True, batch_first=True, bidirectional=True)
        self.rnn3_2 = nn.RNN(input_size=h2_size, hidden_size=h3_size, bias=True, batch_first=True, bidirectional=True)
        self.rnn3_3 = nn.RNN(input_size=h2_size, hidden_size=h3_size, bias=True, batch_first=True, bidirectional=True)

        if bidirectional:
            h3_size = h3_size * 2

        self.rnn4_1 = nn.RNN(input_size=h3_size * 2, hidden_size=h4_size, bias=True, batch_first=True,
                             bidirectional=True)
        self.rnn4_2 = nn.RNN(input_size=h3_size * 2, hidden_size=h4_size, bias=True, batch_first=True,
                             bidirectional=True)

        if bidirectional:
            h4_size = h4_size * 2

        self.rnn5_1 = nn.RNN(input_size=h4_size, hidden_size=h5_size, bias=True, batch_first=True, bidirectional=True)
        self.rnn5_2 = nn.RNN(input_size=h4_size, hidden_size=h5_size, bias=True, batch_first=True, bidirectional=True)

        if bidirectional:
            h5_size = h5_size * 2

        self.lstm = nn.LSTM(input_size=h5_size * 2, hidden_size=h6_size, bias=True, batch_first=True,
                            bidirectional=True)

        if bidirectional:
            h6_size = h6_size * 2
        self.h6_size = h6_size

        self.fc1 = nn.Linear(h6_size*total_length, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # self.fc1 = nn.Linear(h6_size, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, num_classes)

        # self.fc1 = nn.Linear(h6_size, 512)
        # self.fc2 = nn.Linear(512, 128)
        # self.fc3 = nn.Linear(128, num_classes)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x_eyebrow, x_eye , x_nose, x_mouth):
        # L1
        output_eye, hn_eye = self.rnn1_1(x_eye)
        output_eyebrow, hn_eyebrow = self.rnn1_2(x_eyebrow)
        output_nose, hn_nose = self.rnn1_3(x_nose)
        output_mouth, hn_mouth = self.rnn1_4(x_mouth)

        # L2
        output_eye_brow = torch.cat((output_eye, output_eyebrow), dim=2)
        output_eye_brow, hn_eye_brow = self.rnn2_1(output_eye_brow)
        output_nose, hn_nose = self.rnn2_2(output_nose)
        output_mouth, hn_mouth = self.rnn2_3(output_mouth)

        # L3
        output_eye_brow, hn_eye_brow = self.rnn3_1(output_eye_brow)
        output_nose, hn_nose = self.rnn3_2(output_nose)
        output_mouth, hn_mouth = self.rnn3_3(output_mouth)

        # L4
        output_eye_brow_nose = torch.cat((output_eye_brow, output_nose), dim=2)
        output_nose_mouth = torch.cat((output_nose, output_mouth), dim=2)

        output_eye_brow_nose, hn_eye_brow_nose = self.rnn4_1(output_eye_brow_nose)
        output_nose_mouth, hn_nose_mouth = self.rnn4_2(output_nose_mouth)

        # L5
        output_eye_brow_nose, hn_eye_brow_nose = self.rnn5_1(output_eye_brow_nose)
        output_nose_mouth, hn_nose_mouth = self.rnn5_2(output_nose_mouth)

        # L6
        output = torch.cat((output_eye_brow_nose, output_nose_mouth), dim=2)
        output, hn = self.lstm(output)

        # output = output[:, -1:, :]

        output = output.reshape((output.size()[0], -1))

        # FC
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        # output = self.sigmoid(output)

        return output


if __name__ == "__main__":
    device = torch.device('cpu')
    length = 300

    # model = BRNN(input_size=20, hidden_size=30, batch_size=16)
    model_config = {
        'eye_size': 20,
        'eyebrow_size': 20,
        'nose_size': 20,
        'mouth_size': 30,
        'h1_size': 30,
        'h2_size': 30,
        'h3_size': 60,
        'h4_size': 60,
        'h5_size': 90,
        'h6_size': 90,
        'total_length': 300,
        'num_classes': 6
    }

    model = PHRNN(**model_config)
    model = model.to(device)

    input_eye = torch.rand((16, length, 20))
    input_eyebrow = torch.rand((16, length, 20))
    input_nose = torch.rand((16, length, 20))
    input_mouth = torch.rand((16, length, 30))

    # input = torch.rand((16, 30, 20))

    # hidden = model.init_hidden_size()
    #
    output = model(input_eyebrow, input_eye, input_nose, input_mouth)
    print("")
