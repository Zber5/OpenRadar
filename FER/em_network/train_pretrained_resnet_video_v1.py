import torch
import torch.nn as nn
import numpy as np
import random
import time
from dataset import ImglistToTensor, VideoFrameDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from FER.utils import ROOT_PATH, save_to_json
from FER.em_network.utils import model_parameters

from utils import accuracy, device, AverageMeter, dir_path, write_log, save_checkpoint
from models.resnet import resnet18, Classifier, ResNetFull, ResNetFull_Teacher
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


def train(model, data_loader, criterion, optimizer, epoch=0, to_log=None, print_freq=25):
    # create Average Meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_loss = []

    # switch to train mode
    model.classifier.train()
    model.feature.eval()
    # record start time
    start = time.time()

    for i, (inputs, target) in enumerate(data_loader):
        # prepare input and target
        # inputs = torch.permute(inputs, (0, 2, 1, 3, 4))
        inputs = inputs.to(device)
        # target = target.type(torch.LongTensor)
        target = target.long()
        target = target.to(device)

        # measure data loading time
        data_time.update(time.time() - start)

        # zero the parameter gradients
        optimizer.zero_grad()

        # gradient and do SGD step
        output, _ = model(inputs)
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
        for (data, target) in test_loader:
            target = target.long()
            # data = torch.permute(data, (0, 2, 1, 3, 4))
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
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


def denormalize(video_tensor):
    """
    Undoes mean/standard deviation normalization, zero to one scaling,
    and channel rearrangement for a batch of images.
    args:
        video_tensor: a (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    """
    inverse_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inverse_normalize(video_tensor) * 255.).type(torch.uint8).permute(0, 2, 3, 1).numpy()


if __name__ == "__main__":

    config = dict(num_epochs=30,
                  lr=0.0003,
                  lr_step_size=30,
                  lr_decay_gamma=0.2,
                  batch_size=16,
                  num_classes=7,
                  v_num_frames=30,
                  h_num_frames=100,
                  imag_size=224
                  )

    # results dir
    result_dir = "FER/results"
    path = dir_path("Pretrained_ResNet_video_v1_1", result_dir)

    # save training config
    save_to_json(config, path['config'])

    # data path and annotation files
    videos_root = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\'
    annotation_train = os.path.join(videos_root, 'annotations_att_train.txt')
    annotation_test = os.path.join(videos_root, 'annotations_att_test.txt')

    # dataloader
    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_train = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_train,
        num_segments=1,
        frames_per_segment=config['v_num_frames'],
        imagefile_template='frame_{0:012d}.jpg',
        transform=preprocess,
        random_shift=True,
        test_mode=False
    )

    dataset_test = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_test,
        num_segments=1,
        frames_per_segment=config['v_num_frames'],
        imagefile_template='frame_{0:012d}.jpg',
        transform=preprocess,
        random_shift=True,
        test_mode=True
    )

    train_loader = DataLoader(dataset_train, num_workers=4, pin_memory=True, batch_size=config['batch_size'])
    test_loader = DataLoader(dataset_test, num_workers=4, pin_memory=True, batch_size=config['batch_size'])

    # create model
    _structure = resnet18()
    _parameterDir = "C:/Users/Zber/Documents/GitHub/Emotion-FAN/pretrain_model/Resnet18_FER+_pytorch.pth.tar"
    fmodel = model_parameters(_structure, _parameterDir)
    cmodel = Classifier(num_classes=7)
    model = ResNetFull_Teacher(fmodel, cmodel)
    model = model.to(device)

    # print model summary to txt

    # initialize criterion
    criterion = nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step_size'],
                                                   gamma=config['lr_decay_gamma'])

    # defining metrics
    metrics_dic = {
        'loss': [],
        'precision': [],
    }

    best_acc = 0
    for epoch in range(config['num_epochs']):
        train_loss = train(model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch,
                           to_log=path['log'])
        test_loss, acc = test(model, test_loader=test_loader, criterion=criterion, to_log=path['log'])
        if acc >= best_acc:
            best_acc = acc
            save_checkpoint(model.state_dict(), is_best=True, checkpoint=path['dir'], name='model')
            save_checkpoint(model.feature.state_dict(), is_best=True, checkpoint=path['dir'], name='feature')
            save_checkpoint(model.classifier.state_dict(), is_best=True, checkpoint=path['dir'], name='classifier')
        else:
            save_checkpoint(model.state_dict(), is_best=False, checkpoint=path['dir'], name='model', epoch=epoch)
            save_checkpoint(model.feature.state_dict(), is_best=False, checkpoint=path['dir'], name='feature',
                            epoch=epoch)
            save_checkpoint(model.classifier.state_dict(), is_best=False, checkpoint=path['dir'], name='classifier',
                            epoch=epoch)

        lr_scheduler.step()

        metrics_dic['loss'].append(test_loss)
        metrics_dic['precision'].append(acc)

    # print best acc after training
    write_log("<<<<< Best Accuracy = {:.2f} >>>>>".format(best_acc), path['log'])

    # save csv log
    df = pd.DataFrame.from_dict(metrics_dic)
    df.to_csv(path['metrics'], sep='\t', encoding='utf-8')
