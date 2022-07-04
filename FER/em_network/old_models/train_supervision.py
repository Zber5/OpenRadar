import torch
import torch.nn as nn
import numpy as np
import random
import time
from dataset import ConcatDataset, ImglistToTensor, VideoFrameDataset, HeatmapDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from FER.utils import ROOT_PATH, save_to_json
import torch.nn.functional as F

from utils import accuracy, device, AverageMeter, dir_path, write_log, save_checkpoint
from models.c3d import C3D_VIDEO, C3DFusionBaseline
import pandas as pd

import os

os.chdir(ROOT_PATH)

# set seed, make result reporducable
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def train(teacher_model, student_model, data_loader, criterion, optimizer, epoch=0, to_log=None, print_freq=25):
    # create Average Meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_loss = []

    # switch to train mode
    teacher_model.eval()
    student_model.train()
    # record start time
    start = time.time()

    for i, conc_data in enumerate(data_loader):
        h_data, v_data = conc_data
        azi, ele, targets = h_data
        v_inputs, _ = v_data

        # prepare input and target
        azi = azi.to(device, dtype=torch.float)
        ele = ele.to(device, dtype=torch.float)
        v_inputs = torch.permute(v_inputs, (0, 2, 1, 3, 4)).to(device)
        targets = targets.to(device, dtype=torch.long)

        # measure data loading time
        data_time.update(time.time() - start)

        # zero the parameter gradients
        optimizer.zero_grad()

        # gradient and do SGD step
        pseudo_targets = teacher_model(v_inputs)
        outputs = student_model(azi, ele)
        loss = criterion(outputs, pseudo_targets, targets)

        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
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


def _make_criterion(alpha=0.5, T=4.0, mode='cse'):
    def criterion(outputs, targets, labels):
        if mode == 'cse':
            _p = F.log_softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = -torch.mean(torch.sum(_q * _p, dim=1))
        elif mode == 'mse':
            _p = F.softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = nn.MSELoss()(_p, _q) / 2
        else:
            raise NotImplementedError()

        _soft_loss = _soft_loss * T * T
        _hard_loss = F.cross_entropy(outputs, labels)
        loss = alpha * _soft_loss + (1. - alpha) * _hard_loss
        return loss

    return criterion


if __name__ == "__main__":

    config = dict(num_epochs=60,
                  lr=0.001,
                  lr_step_size=20,
                  lr_decay_gamma=0.2,
                  batch_size=16,
                  num_classes=7,
                  v_num_frames=30,
                  h_num_frames=100,
                  imag_size=224,
                  weight_alpha=0.7,
                  softmax_temperature=16.0,
                  loss_mode='cse')

    # results dir
    result_dir = "../../results"
    path = dir_path("C3D_Supervision", result_dir)

    # save training config
    save_to_json(config, path['config'])

    # load data
    videos_root = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\'
    v_train_ann = os.path.join(videos_root, 'annotations_att_train.txt')
    v_test_ann = os.path.join(videos_root, 'annotations_att_test.txt')

    heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"
    h_train_ann = os.path.join(heatmap_root, "heatmap_annotation_train.txt")
    h_test_ann = os.path.join(heatmap_root, "heatmap_annotation_test.txt")

    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # video datasets
    video_train = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=v_train_ann,
        num_segments=1,
        frames_per_segment=config['v_num_frames'],
        imagefile_template='frame_{0:012d}.jpg',
        transform=preprocess,
        random_shift=False,
        test_mode=False
    )

    video_test = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=v_test_ann,
        num_segments=1,
        frames_per_segment=config['v_num_frames'],
        imagefile_template='frame_{0:012d}.jpg',
        transform=preprocess,
        random_shift=False,
        test_mode=True
    )

    # heatmap datasets
    heatmap_train = HeatmapDataset(heatmap_root, h_train_ann)
    heatmap_test = HeatmapDataset(heatmap_root, h_test_ann)

    dataset_train = ConcatDataset(heatmap_train, video_train)

    train_loader = DataLoader(dataset_train, num_workers=4, pin_memory=True, batch_size=config['batch_size'])
    test_loader = DataLoader(heatmap_test, num_workers=4, pin_memory=True, batch_size=config['batch_size'])

    # create model
    teacher_model = C3D_VIDEO(sample_size=config['imag_size'], sample_duration=config['v_num_frames'], num_classes=config['num_classes'])
    checkpoint = os.path.join(result_dir, "C3D_Video_att_20211205-002110", 'best_model.pt')
    assert os.path.exists(checkpoint), 'Error: no checkpoint directory found!'

    teacher_model.load_state_dict(torch.load(checkpoint))
    teacher_model = teacher_model.to(device)

    student_model = C3DFusionBaseline(sample_duration=config['h_num_frames'], num_classes=config['num_classes'])
    student_model = student_model.to(device)

    # initialize critierion and optimizer
    criterion = _make_criterion(alpha=config['weight_alpha'], T=config['softmax_temperature'], mode=config['loss_mode'])
    test_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=config['lr'])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step_size'], gamma=config['lr_decay_gamma'])

    metrics_dic = {
        'loss': [],
        'precision': [],
    }

    best_acc = 0
    for epoch in range(config['num_epochs']):
        train_loss = train(teacher_model, student_model, data_loader=train_loader, criterion=criterion,
                           optimizer=optimizer, epoch=epoch,
                           to_log=path['log'])
        test_loss, acc = test(student_model, test_loader=test_loader, criterion=test_criterion, to_log=path['log'])
        if acc >= best_acc:
            best_acc = acc
            save_checkpoint(student_model.state_dict(), is_best=True, checkpoint=path['dir'])
        else:
            save_checkpoint(student_model.state_dict(), is_best=False, checkpoint=path['dir'])

        lr_scheduler.step()

        metrics_dic['loss'].append(test_loss)
        metrics_dic['precision'].append(acc)

    # print best acc after training
    write_log("<<<<< Best Accuracy = {:.2f} >>>>>".format(best_acc), path['log'])

    # save csv log
    df = pd.DataFrame.from_dict(metrics_dic)
    df.to_csv(path['metrics'], sep='\t', encoding='utf-8')
