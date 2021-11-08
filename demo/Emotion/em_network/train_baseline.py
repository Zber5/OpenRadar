import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder

import time
from demo.Emotion.em_network.utils import AverageMeter, accuracy, dir_path, SensorDataSet, normalise_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set seed, make result reporducable
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def data_loader(x, y, batch_size=50):
    dataset = SensorDataSet(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


class LeNet(nn.Module):
    def __init__(self, in_channel, out1_channel, out2_channel, fc, out_classes, kernel_size, flatten_factor,
                 padding=[0, 0]):
        super(LeNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channel, out1_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[0])),
            # nn.BatchNorm2d(out1_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(out1_channel, out2_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[1])),
            # nn.BatchNorm2d(out2_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(flatten_factor * out2_channel, fc),
            nn.ReLU(inplace=True),
            nn.Linear(fc, out_classes),
        )

    def forward(self, x):
        x = self.features(x)
        out1 = x.view(x.size(0), -1)
        x = self.classifier(out1)
        out2 = F.softmax(x, dim=1)
        return out2


def write_log(str, path_to_log):
    if isinstance(path_to_log, dict):
        for key in path_to_log:
            log = open(path_to_log[key], 'a')
            log.write(str)
            log.close()

    else:
        log = open(path_to_log, 'a')
        log.write(str)
        log.close()


def train(model, data_loader, criterion, optimizer, epoch=0, to_log=None, print_freq=100):
    # create Average Meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_loss = []

    # switch to train mode
    model.train()
    # record start time
    start = time.time()

    for i, (inputs, target) in enumerate(data_loader):
        # prepare input and target
        inputs = inputs.to(device)
        # target = target.type(torch.LongTensor)
        target = target.long()
        target = target.to(device)


        # measure data loading time
        data_time.update(time.time() - start)

        # zero the parameter gradients
        optimizer.zero_grad()

        # gradient and do SGD step
        output = model(inputs)
        loss = criterion(output, target)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

        # print training info
        if i % print_freq == 0:
            str = ('Epoch: [{0}][{1}/{2}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:3.3f} ({top1.avg:3.3f})\t'
                   'Prec@5 {top5.val:3.3f} ({top5.avg:3.3f})'.format(
                epoch, i, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            print(str)

            if to_log is not None:
                write_log(str + '\n', to_log)

    return train_loss


def test(model, test_loader, criterion, to_log=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.sampler)
    test_loss *= test_loader.batch_size
    acc = 100. * correct / len(test_loader.sampler)
    format_str = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.sampler), acc
    )
    print(format_str)
    if to_log is not None:
        write_log(format_str, to_log)
    return test_loss.item(), acc


if __name__ == "__main__":

    N_EPOCHS = 100
    LR = 0.005
    BATCH_SIZE = 50
    model_para = {
        'in_channel': 8,
        'out1_channel': 20,
        'out2_channel': 50,
        'fc': 100,
        'out_classes': 7,
        # 'kernel_size': 5,
        'kernel_size': 10,
        # 'flatten_factor': 9,
        'flatten_factor': 30,
    }

    # results dir
    result_dir = "/results"

    # load data
    # df_x = np.load('C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff_segment_x.npy')
    # df_y = np.load('C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff_segment_y.npy')


    df_x = np.load('C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff_x.npy')
    df_y = np.load('C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff_y.npy')

    # normalization
    df_x = normalise_data(df_x)
    df_x = np.expand_dims(df_x, axis=2)
    # df_y_eye = np_to_eye(df_y, num_class=7)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=25, stratify=df_y)

    train_loader = data_loader(x_train, y_train, batch_size=BATCH_SIZE)
    test_loader = data_loader(x_test, y_test, batch_size=BATCH_SIZE)

    # log path
    path = dir_path("emotion_3s_diff_segment", result_dir)

    # create model
    model = LeNet(**model_para)
    model = model.to(device)

    # initialize critierion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        train_loss = train(model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch,
                           to_log=path['log'])
        test_loss, acc = test(model, test_loader=test_loader, criterion=criterion, to_log=path['log'])
