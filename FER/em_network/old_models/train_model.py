import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import random
import time

from utils import twin_sliding_window, sensor_imag_data_loader, device, AverageMeter, dir_path, write_log
from models.model import Autoencoder

# set seed, make result reporducable
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def rmse(y, y_hat):
    """Compute root mean squared error"""
    length = y.size()[1]
    return torch.sqrt(torch.mean((y - y_hat).pow(2)))/length


class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.record = 0

    def __len__(self):
        return self.X.__len__() - (self.seq_len - 1)

    def __getitem__(self, index):
        return (self.X[index:index + self.seq_len], self.y[index + self.seq_len - 1])


def format_dataset(sensor, imag, label, imag_window=6, sensor_window=20, imag_step=3, sensor_step=10):
    num_data = np.shape(sensor)[0]
    sensor_channels = np.shape(sensor)[1]
    imag_channels = np.shape(imag)[1]
    sensor_length = np.shape(sensor)[2]
    imag_length = np.shape(imag)[2]

    num_segment = imag_length // imag_step

    x = np.zeros((num_segment * num_data, sensor_channels, sensor_window))
    y = np.zeros((num_segment * num_data, imag_channels))
    l = np.zeros((num_segment * num_data))
    index = 0
    for i in range(num_data):
        label_content = l[i]
        for imag_idx, sensor_idx in twin_sliding_window(imag_length, sensor_length, imag_window, sensor_window,
                                                        imag_step, sensor_step):
            sensor_start, sensor_end = sensor_idx
            imag_start, imag_end = imag_idx
            x[index] = sensor[i, :, sensor_start:sensor_end]
            imag_mean = np.mean(imag[i, :, imag_start:imag_end], axis=1)
            y[index] = imag_mean
            l[index] = label_content
            index += 1

    return x, y, l


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
        # target = target.long()
        target = target.to(device)

        # measure data loading time
        data_time.update(time.time() - start)

        # zero the parameter gradients
        optimizer.zero_grad()

        # gradient and do SGD step
        output = model(inputs)
        loss = criterion(output, target)
        mse = rmse(output, target)

        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        # top1.update(prec1.item(), inputs.size(0))
        top1.update(0, 1)
        # top5.update(prec5.item(), inputs.size(0))
        top5.update(0, 1)

        # measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

        # print training info
        if i % print_freq == 0:
            str = ('Epoch: [{0}][{1}/{2}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'MSE {mse:3.5f}'.format(
                epoch, i, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, mse=mse))
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
            # target = target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)  # sum up batch loss
            mse = rmse(output, target)
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.sampler)
    test_loss *= test_loader.batch_size
    # acc = 100. * correct / len(test_loader.sampler)
    acc = 0
    format_str = 'Test set: Average loss: {:.4f}, MSE: ({:3.5f})\n'.format(test_loss, mse)
    print(format_str)
    if to_log is not None:
        write_log(format_str, to_log)
    return test_loss.item(), acc


def scale_range(input, min=0, max=1):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input


if __name__ == "__main__":

    N_EPOCHS = 100
    LR = 0.0005
    BATCH_SIZE = 16
    # model_para = {
    #     'in_channel': 8,
    #     'out1_channel': 20,
    #     'out2_channel': 50,
    #     'fc': 100,
    #     'out_classes': 7,
    #     # 'kernel_size': 5,
    #     'kernel_size': 10,
    #     # 'flatten_factor': 9,
    #     'flatten_factor': 30,
    # }

    # results dir
    result_dir = "../../results"

    # load data
    sensor_data_path = '../../data/sensor_b8_c5_x.npy'
    flm_score_path = '../../data/flm_score.npy'
    label_path = '../../data/sensor_b8_c5_y.npy'
    sensor = np.load(sensor_data_path)
    imag = np.load(flm_score_path)
    label = np.load(label_path)

    # formatting data
    imag = imag.transpose((0, 2, 1))
    x, y, label = format_dataset(sensor, imag, label)
    x = np.expand_dims(x, axis=2)
    # y = np.expand_dims(y, axis=2)

    # preprocessing data to min-max scale
    x = scale_range(x)
    y = scale_range(y)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25, stratify=label)

    train_loader = sensor_imag_data_loader(x_train, y_train, batch_size=BATCH_SIZE)
    test_loader = sensor_imag_data_loader(x_test, y_test, batch_size=BATCH_SIZE)

    # log path
    path = dir_path("sensor_imag_basic", result_dir)

    # create model
    model = Autoencoder()
    model = model.to(device)

    # initialize critierion and optimizer
    # could add weighted loss e.g. pos_weight = torch.ones([64])
    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        train_loss = train(model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch,
                           to_log=path['log'])
        test_loss, acc = test(model, test_loader=test_loader, criterion=criterion, to_log=path['log'])

    # save final model
    torch.save(model.state_dict(), path['model'])
