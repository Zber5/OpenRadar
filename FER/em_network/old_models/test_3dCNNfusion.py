import torch
from sklearn.model_selection import train_test_split
import numpy as np
import random
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from utils import device, AverageMeter, write_log, \
    senor_heatmap_label_data_loader, accuracy, normalise_data
from models.c3d import C3DFusionBaseline
import seaborn as sns

import os

os.chdir('//')

# set seed, make result reporducable
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


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

    for i, (inputs, _, target) in enumerate(data_loader):
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
        for (data, _, target) in test_loader:
            target = target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
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
    # path = dir_path("sensor_heatmap_3dcnn", result_dir)

    # create model
    model = C3DFusionBaseline(sample_duration=num_frames, num_classes=num_classes)
    model_path = "FER/results/sensor_heatmap_3dcnn_fusion20211030-123808/model.ckpt"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # test
    azi_test = azi_test.astype('float32')
    azi_test = torch.from_numpy(azi_test).to(device)

    ele_test = ele_test.astype('float32')
    ele_test = torch.from_numpy(ele_test).to(device)

    output = model(azi_test, ele_test)
    y_pred = output.argmax(dim=1, keepdim=True)
    y_pred = y_pred.numpy().flatten()
    y_test = label_test

    print(classification_report(y_test, y_pred, target_names=emotion_list))
    cm = confusion_matrix(y_test, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)  # annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(emotion_list)
    ax.yaxis.set_ticklabels(emotion_list)
    plt.show()

    print()


