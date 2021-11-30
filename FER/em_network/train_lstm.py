from models.mmwave_model import mmwave_lstm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import random
import time

from utils import twin_sliding_window, device, write_log
from torch.utils.data import DataLoader
from utils import AverageMeter, accuracy, dir_path, SensorDataSet

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


class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.record = 0

    def __len__(self):
        return self.X.__len__() - (self.seq_len - 1)

    def __getitem__(self, index):
        return (self.X[index:index + self.seq_len], self.y[index + self.seq_len - 1])


def format_dataset(sensor, imag, label, imag_window=9, sensor_window=30, imag_step=3, sensor_step=10):
    num_data = np.shape(sensor)[0]
    sensor_channels = np.shape(sensor)[1]
    sensor_bins = np.shape(sensor)[3]
    imag_channels = np.shape(imag)[1]
    sensor_length = np.shape(sensor)[2]
    imag_length = np.shape(imag)[2]

    intensity_unit = 0.1

    num_segment = (imag_length - imag_window) // imag_step + 1

    x = np.zeros((num_segment * num_data, sensor_channels, sensor_window, sensor_bins))
    y = np.zeros((num_segment * num_data, imag_channels))
    l = np.zeros((num_segment * num_data))
    index = 0
    for i in range(num_data):
        label_content = label[i]
        for imag_idx, sensor_idx in twin_sliding_window(imag_length, sensor_length, imag_window, sensor_window,
                                                        imag_step, sensor_step):
            sensor_start, sensor_end = sensor_idx
            imag_start, imag_end = imag_idx
            x[index] = sensor[i, :, sensor_start:sensor_end]
            # imag_mean = np.mean(imag[i, :, imag_start:imag_end], axis=1)
            imag_diff = imag[i, :, imag_end] - imag[i, :, imag_start]
            cond = imag_diff > intensity_unit
            binary_label = cond.astype(int)
            y[index] = binary_label
            l[index] = label_content
            index += 1

    return x, y, l


def loss_fn(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    bceloss_fn = nn.BCELoss(size_average=False)
    BCE = bceloss_fn(recon_x, x)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


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
        inputs = inputs.permute(2, 0, 1)
        inputs = inputs.to(device)
        # target = target.type(torch.LongTensor)
        target = target.long()
        target = target.to(device)

        # measure data loading time
        data_time.update(time.time() - start)

        # zero the parameter gradients
        optimizer.zero_grad()

        # gradient and do SGD step
        model.batch_size = len(target)
        model.hidden = model.init_hidden()
        output, _ = model(inputs)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
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
            data = data.permute(2, 0, 1)
            data, target = data.to(device), target.to(device)
            model.batch_size = len(target)
            model.hidden = model.init_hidden()
            output, hidden = model(data)
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


def scale_range(input, min=0, max=1):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input


def normalise_data(data):
    nor_data = (data - np.mean(data)) / np.std(data)
    return nor_data


if __name__ == "__main__":

    N_EPOCHS = 100
    LR = 0.0005
    BATCH_SIZE = 64

    class_names = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    # results dir
    result_dir = "FER/results"

    # load data
    sensor_data_path = 'FER/data/sensor_b8r3_c5_x.npy'
    flm_score_path = 'FER/data/aus_rf.npy'
    label_path = 'FER/data/sensor_b8r3_c5_y.npy'

    sensor_data_path_1 = 'FER/data/sensor_b8r3_c5_x_s40_e80.npy'
    flm_score_path_1 = 'FER/data/aus_rf_s40_e80.npy'
    label_path_1 = 'FER/data/sensor_b8r3_c5_y_s40_e80.npy'
    sensor = np.load(sensor_data_path)
    label = np.load(label_path)

    sensor_1 = np.load(sensor_data_path_1)
    label_1 = np.load(label_path_1)

    x = np.concatenate((sensor, sensor_1), axis=0)
    y = np.concatenate((label, label_1), axis=0)

    x = np.transpose(x, (0, 1, 3, 2))
    x = x.reshape((-1, 36, 299))
    # x = x.transpose((2, 0, 1))
    # preprocessing data to min-max scale
    x = scale_range(x)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25, stratify=y)

    train_loader = data_loader(x_train, y_train, batch_size=BATCH_SIZE)
    test_loader = data_loader(x_test, y_test, batch_size=BATCH_SIZE)

    # log path
    path = dir_path("sensor_lstm", result_dir)

    # model parameters
    input_size, hidden_size, num_classes = 36, 64, 6
    model = mmwave_lstm(input_size, hidden_size, BATCH_SIZE, num_classes=num_classes)
    # create model
    model = model.to(device)

    # initialize critierion and optimizer
    # could add weighted loss e.g. pos_weight = torch.ones([64])
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    metrics_dic = {
        'loss': [],
        'precision': []
    }

    for epoch in range(N_EPOCHS):
        train_loss = train(model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch,
                           to_log=path['log'])
        test_loss, pre = test(model, test_loader=test_loader, criterion=criterion, to_log=path['log'])

        metrics_dic['loss'].append(test_loss)
        metrics_dic['precision'].append(pre)

    # save final model
    torch.save(model.state_dict(), path['model'])

    # save metrics dic
