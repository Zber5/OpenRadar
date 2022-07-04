import torch
import torch.nn as nn
import numpy as np
import random
import time

from utils import device, AverageMeter, dir_path, write_log, accuracy
from models.c3d import C3DFusionBaseline
from dataset import HeatmapDataset
from torch.utils.data import DataLoader
import os
import pandas as pd
from FER.utils import ROOT_PATH
os.chdir(ROOT_PATH)

# set seed, make result reporducable
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def train(model, data_loader, criterion, optimizer, epoch=0, to_log=None, print_freq=10):
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
        # prepare input and target to device
        azi = azi.to(device, dtype=torch.float)
        ele = ele.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.long)

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
            # prepare input and target to device
            azi = azi.to(device, dtype=torch.float)
            ele = ele.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.long)

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


if __name__ == "__main__":

    N_EPOCHS = 60
    LR = 0.0003
    BATCH_SIZE = 16
    num_classes = 7
    num_frames = 100

    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust', 'Neutral']

    # results dir
    result_dir = "../../results"

    # heatmap root dir
    heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"

    # annotation dir
    annotation_train = os.path.join(heatmap_root, "heatmap_annotation_train.txt")
    annotation_test = os.path.join(heatmap_root, "heatmap_annotation_test.txt")

    # load data
    dataset_train = HeatmapDataset(heatmap_root, annotation_train)
    dataset_test = HeatmapDataset(heatmap_root, annotation_test)
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # log path
    path = dir_path("sensor_heatmap_3dcnn_fusion_v2", result_dir)

    # create model
    model = C3DFusionBaseline(sample_duration=num_frames, num_classes=num_classes)
    model = model.to(device)

    # initialize critierion and optimizer
    # could add weighted loss e.g. pos_weight = torch.ones([64])
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

    metrics_dic = {
        'loss': [],
        'precision': []
    }

    best_acc = 0
    for epoch in range(N_EPOCHS):
        train_loss = train(model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch,
                           to_log=path['log'])
        test_loss, acc = test(model, test_loader=test_loader, criterion=criterion, to_log=path['log'])

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), path['model'])

        lr_scheduler.step()

        metrics_dic['loss'].append(test_loss)
        metrics_dic['precision'].append(acc)

    # save csv log
    df = pd.DataFrame.from_dict(metrics_dic)
    df.to_csv(path['metrics'], sep='\t', encoding='utf-8')
