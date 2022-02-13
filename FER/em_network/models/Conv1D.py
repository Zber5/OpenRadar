import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class TimeDistributed1D(nn.Module):
    def __init__(self, module):
        super(TimeDistributed1D, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        n, t = x.size(0), x.size(1)
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(n * t, x.size(2), x.size(3))
        y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(n, t, y.size(1), y.size(2))
        return y


class SubNet1D(nn.Module):
    def __init__(self, num_channel=2):
        super(SubNet1D, self).__init__()

        self.group1 = nn.Sequential(
            nn.Conv1d(num_channel, 16, kernel_size=9, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2))

        self.group2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2))

        self.group3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2))

        self.group4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2))

        self.group5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2))

    def forward(self, x):
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes=7, input_channel=128 * 13):
        super(Classifier, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_channel, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class LandmarkLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size, num_layers=2):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(LandmarkLSTM, self).__init__()
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


class LandmarkNet(nn.Module):
    def __init__(self, subnet=SubNet1D, classifier=Classifier, num_channel=2, num_classes=7):
        super(LandmarkNet, self).__init__()
        self.features = TimeDistributed1D(subnet(num_channel))
        self.classifier = classifier(num_classes)

    def forward(self, x):
        x = self.features(x)
        # x = torch.mean(x, dim=1) # v1
        x, _ = torch.max(x, dim=1)  # v2
        x = x.view((x.size(0), -1))
        x = self.classifier(x)
        return x


class LandmarkNet_LSTM(nn.Module):
    def __init__(self, subnet=SubNet1D, classifier=Classifier, num_channel=2, num_classes=7, input_size=468 * 2,
                 hidden_size=468 * 2, batch_size=16):
        super(LandmarkNet_LSTM, self).__init__()

        self.num_landmarks = 468
        self.lstm = LandmarkLSTM(input_size=input_size, hidden_size=hidden_size, batch_size=batch_size)
        self.features = subnet(num_channel)
        self.classifier = classifier(num_classes)

    def forward(self, x):
        x = self.lstm(x)
        x = x.view((x.size(0), -1, self.num_landmarks))

        x = self.features(x)
        # x = torch.mean(x, dim=1) # v1
        # x, _ = torch.max(x, dim=1)  # v2
        x = x.view((x.size(0), -1))
        x = self.classifier(x)
        return x


class SubNet_2D(nn.Module):
    def __init__(self, num_channel=1):
        super(SubNet_2D, self).__init__()

        self.group1 = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=(8, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)))

        self.group2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)))

        self.group3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(8, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(5, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )

    def forward(self, x):
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)

        return x


class LandmarkNet_2D(nn.Module):
    def __init__(self, subnet=SubNet_2D, classifier=Classifier, num_channel=2, num_classes=7):
        super(LandmarkNet_2D, self).__init__()
        self.features = subnet(num_channel)
        self.classifier = classifier(num_classes, input_channel=12*3*128)

    def forward(self, x):
        x = self.features(x)
        x = x.view((x.size(0), -1))
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    device = torch.device('cuda')
    # model = LandmarkNet(subnet=SubNet1D, classifier=Classifier, num_channel=2, num_classes=7)
    # model = LandmarkNet_LSTM(subnet=SubNet1D, classifier=Classifier, num_channel=2, num_classes=7, input_size=468 * 2,
    #                          hidden_size=468 * 2, batch_size=16)

    model = LandmarkNet_2D(subnet=SubNet_2D, classifier=Classifier, num_channel=2, num_classes=7)
    model = model.to(device)

    input1 = torch.randn(16, 30, 2, 468)
    # input1 = torch.reshape(input1, (16, 30, -1))
    input1 = torch.permute(input1, (0, 2, 3, 1))
    input1 = input1.to(device)

    output = model(input1)
    print(output.view((output.size(0), -1)).size())
