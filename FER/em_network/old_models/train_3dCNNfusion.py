import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import random
import time

from utils import twin_sliding_window, device, AverageMeter, dir_path, write_log, \
    senor_heatmap_label_data_loader, accuracy, normalise_data
from models.autoencoder import EMOEncode, EMODecode
from models.c3d import C3DFusionBaseline

import os

os.chdir('../../../demo/')

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
    return torch.sqrt(torch.mean((y - y_hat).pow(2))) / length


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    epsilon = torch.tensor([1e-7]).to(device)
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision.item()


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    epsilon = torch.tensor([1e-7]).to(device)
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall.item()


def fbeta_score(y_true, y_pred, beta=1):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if torch.sum(torch.round(torch.clip(y_true, 0, 1))) == 0:
        return 0
    epsilon = torch.tensor([1e-7]).to(device)
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + epsilon)
    return fbeta_score


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


def train(model, data_loader, criterion, optimizer, epoch=0, to_log=None, print_freq=5):
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

    for i, (azi, ele, target) in enumerate(data_loader):
        # prepare input and target
        azi = azi.to(device)
        ele = ele.to(device)
        # target = target.type(torch.LongTensor)
        target = target.long()
        target = target.to(device)

        # measure data loading time
        data_time.update(time.time() - start)

        # zero the parameter gradients
        optimizer.zero_grad()

        # gradient and do SGD step
        output = model(azi, ele)
        loss = criterion(output, target)

        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), data_loader.batch_size)
        top1.update(prec1.item(), data_loader.batch_size)
        top5.update(prec5.item(), data_loader.batch_size)

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
        for (azi, ele, target) in test_loader:
            azi, ele, target = azi.to(device), ele.to(device), target.to(device)
            target = target.long()
            output = model(azi, ele)
            loss = criterion(output, target)
            test_loss += loss
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


class EMONet(nn.Module):
    def __init__(self, n_class):
        super(EMONet, self).__init__()

        self.encode = EMOEncode()
        self.decode = EMODecode(n_class)
        self.fc1 = nn.Linear(192, 32)
        self.fc2 = nn.Linear(192, 32)
        self.fc3 = nn.Linear(32, 192)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        # z = mu + std
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        h = self.encode(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        z = self.decode(z)
        return z, mu, logvar


if __name__ == "__main__":

    N_EPOCHS = 30
    LR = 0.0005
    BATCH_SIZE = 32
    num_classes = 6
    num_frames = 300

    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    # results dir
    result_dir = "../../results"

    # load data

    azi_data_path = "../../data/Heatmap_D0_S1_L0_B4-14_I0-80_azi.npy"
    ele_data_path = "../../data/Heatmap_D0_S1_L0_B4-14_I0-80_ele.npy"
    # label_path = 'demo/Emotion/data/sensor_b8r3_c5_y.npy'
    # label_path_1 = 'demo/Emotion/data/sensor_b8r3_c5_y_s40_e80.npy'

    azi = np.load(azi_data_path)
    ele = np.load(ele_data_path)

    label = np.zeros((len(emotion_list), 80))
    for i in range(len(label)):
        label[i] = i
    label = label.flatten()

    # expand dims
    azi = np.expand_dims(azi, axis=1)
    ele = np.expand_dims(ele, axis=1)

    # normalize
    azi = normalise_data(azi)
    ele = normalise_data(ele)

    # split data
    azi_train, azi_test, ele_train, ele_test, label_train, label_test = train_test_split(azi, ele, label, test_size=0.2,
                                                                                         random_state=25,
                                                                                         stratify=label)

    train_loader = senor_heatmap_label_data_loader(azi_train, ele_train, label_train, batch_size=BATCH_SIZE)
    test_loader = senor_heatmap_label_data_loader(azi_test, ele_test, label_test, batch_size=np.shape(azi_test)[0])

    # log path
    path = dir_path("sensor_heatmap_3dcnn_fusion", result_dir)

    # create model
    model = C3DFusionBaseline(sample_duration=num_frames, num_classes=num_classes)
    model = model.to(device)

    # initialize critierion and optimizer
    # could add weighted loss e.g. pos_weight = torch.ones([64])
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    metrics_dic = {
        'loss': [],
        'precision': [],
        'recall': []
    }

    for epoch in range(N_EPOCHS):
        train_loss = train(model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch,
                           to_log=path['log'])
        test_loss, acc = test(model, test_loader=test_loader, criterion=criterion, to_log=path['log'])

        lr_scheduler.step()

        metrics_dic['loss'].append(test_loss)
        metrics_dic['precision'].append(acc)
        # metrics_dic['recall'].append(rec)

    # save final model
    torch.save(model.state_dict(), path['model'])

    # save metrics dic
