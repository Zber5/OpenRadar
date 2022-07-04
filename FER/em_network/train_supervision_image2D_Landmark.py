import torch
import torch.nn as nn
import numpy as np
import random
import time
from dataset import ConcatDataset, ImglistToTensor, VideoFrameDataset, HeatmapDataset, LandmarkDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from FER.utils import ROOT_PATH, save_to_json
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from utils import accuracy, device, AverageMeter, dir_path, write_log, save_checkpoint
from models.resnet import resnet18, Classifier, ResNetFull_Teacher
from models.Conv2D import ImageSingle_Student, ImageSingle_v1, ImageDualNet_Single_v1, ImageNet_Large_v1
from models.ConvLSTM import ConvLSTMFull_ME
from metric_losses import NPairLoss_CrossModal, NPairLoss, NPairLoss_CrossModal_Weighted
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from models.Conv1D import LandmarkNet, Classifier, SubNet1D, SubNet_2D, LandmarkNet_2D

from FER.utils import MapRecord

# torch.autograd.set_detect_anomaly(True)

import os

os.chdir(ROOT_PATH)

# set seed, make result reporducable
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))


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


#
# def at(x):
#     if len(x.shape) != 5:
#         return F.normalize(x.pow(2).mean((2, 3)).view(x.size(0), -1))
#     else:
#         return F.normalize(x.pow(2).mean((1, 3, 4)).view(x.size(0), -1))
#
#
# def at_loss(x, y):
#     return (at(x) - at(y)).pow(2).mean()
#
#
# def pair_loss(x, y):
#     pdist = nn.PairwiseDistance()
#     x = at(x)
#     y = at(y)
#     diff = pdist(x, y).mean()
#     return diff

def pair_loss(x, y):
    pdist = nn.PairwiseDistance()
    diff = pdist(x, y).mean()
    return diff


class TFblock(nn.Module):
    def __init__(self, dim_in=521, dim_inter=128, dim_out=936):
        super(TFblock, self).__init__()
        self.encoder = nn.Linear(dim_in, dim_inter)
        self.decoder1 = nn.Linear(dim_inter, dim_inter * 2)
        self.decoder2 = nn.Linear(dim_inter * 2, dim_out)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        # self.bn1 = nn.BatchNorm1d(dim_inter)
        # self.bn2 = nn.BatchNorm1d(dim_inter * 2)

    def forward(self, x):
        x = self.encoder(x)
        # x = self.relu1(x)
        x = self.decoder1(x)
        # x = self.relu2(x)
        x = self.decoder2(x)

        return x


class Classifier_Transformer(nn.Module):
    def __init__(self, input_dim=512, num_classes=7):
        super(Classifier_Transformer, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 512),
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


def train(teacher_model, student_model, data_loader, criterion, optimizer, epoch=0, to_log=None, print_freq=25):
    # create Average Meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    cls_losses = AverageMeter()
    kd_losses = AverageMeter()
    npair_losses = AverageMeter()
    npaircross_losses = AverageMeter()
    train_loss = []

    # switch to train mode
    student_model.train()
    tf_block.train()
    student_classifier.train()

    # record start time
    start = time.time()

    criterion_cls = criterions['criterionCls']
    criterion_kd = criterions['criterionKD']

    for i, conc_data in enumerate(data_loader):
        h_data, v_data = conc_data
        azi, ele, targets = h_data
        v_inputs, _ = v_data

        # prepare input and target
        # azi = torch.permute(azi, (0, 2, 1, 3, 4))
        # ele = torch.permute(ele, (0, 2, 1, 3, 4))
        azi = azi.to(device, dtype=torch.float)
        ele = ele.to(device, dtype=torch.float)
        # v_inputs = torch.permute(v_inputs, (0, 2, 1, 3, 4)).to(device)
        v_inputs = v_inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)

        # measure data loading time
        data_time.update(time.time() - start)

        # zero the parameter gradients
        optimizer.zero_grad()

        _, g_s = student_model(azi, ele)
        s1, s2, s3, s4 = g_s
        # s_fmap = s4
        s_fmap = torch.squeeze(avgpool(s4))
        # s_fmap = F.normalize(s_fmap)
        s_fmap = tf_block(s_fmap)
        # s_fmap_avg = F.normalize(s_fmap)
        # t_fmap_avg = F.normalize(torch.reshape(v_inputs, (-1, 936)))

        s_fmap_avg = s_fmap
        t_fmap_avg = torch.reshape(v_inputs, (-1, 936))

        mse = criterion_kd(s_fmap_avg, t_fmap_avg)
        # mse = pair_loss(s_fmap_avg, t_fmap_avg)

        # outputs = student_classifier(s_fmap_avg)
        # outputs = student_classifier(s_fmap)
        s4 = avgpool(s4)
        s4_input = s4.view(s4.size(0), -1)
        outputs = student_classifier(s4_input)
        cls_loss = criterion_cls(outputs, targets)
        loss = cls_loss + config['mse_weight'] * mse
        # mse = cls_loss
        # loss = cls_loss

        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        cls_losses.update(cls_loss.item(), data_loader.batch_size)
        kd_losses.update(mse.item(), data_loader.batch_size)
        losses.update(loss.item(), data_loader.batch_size)
        top1.update(prec1.item(), data_loader.batch_size)
        top5.update(prec5.item(), data_loader.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

        # print training info
        if i % print_freq == 0:
            str = ('Epoch: [{0}][{1}/{2}]\t'
                   'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Cls:{cls_losses.val:.4f} ({cls_losses.avg:.4f})  '
                   'KD:{kd_losses.val:.4f} ({kd_losses.avg:.4f})  '
                   'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1: {top1.val:3.3f} ({top1.avg:3.3f})\t'
                   'Prec@5: {top5.val:3.3f} ({top5.avg:3.3f})'.format(
                epoch, i, len(data_loader), batch_time=batch_time, cls_losses=cls_losses, kd_losses=kd_losses,
                npair_losses=npair_losses, npaircross_losses=npaircross_losses, data_time=data_time, loss=losses,
                top1=top1, top5=top5))
            print(str)

            if to_log is not None:
                write_log(str + '\n', to_log)

    return losses.avg, cls_losses.avg, kd_losses.avg


def test(s_model, t_model, test_loader, criterion, to_log=None):
    s_model.eval()
    student_classifier.eval()
    tf_block.eval()
    test_loss = 0
    kd_test_loss = 0
    correct = 0
    npair_loss = 0
    # acc = 0
    with torch.no_grad():
        for i, conc_data in enumerate(test_loader):
            h_data, v_data = conc_data
            azi, ele, target = h_data
            v_inputs, _ = v_data

            # prepare input and target
            # azi = torch.permute(azi, (0, 2, 1, 3, 4))
            # ele = torch.permute(ele, (0, 2, 1, 3, 4))
            azi = azi.to(device, dtype=torch.float)
            ele = ele.to(device, dtype=torch.float)
            # v_inputs = torch.permute(v_inputs, (0, 2, 1, 3, 4)).to(device)
            v_inputs = v_inputs.to(device)
            target = target.to(device, dtype=torch.long)

            # appley motion extractor before feed into student model
            # azi, ele = extractor(azi, ele)

            _, g_s = student_model(azi, ele)
            s1, s2, s3, s4 = g_s

            # s_fmap = s4
            s_fmap = torch.squeeze(avgpool(s4))
            s_fmap = tf_block(s_fmap)

            # s_fmap_avg = F.normalize(s_fmap)
            # t_fmap_avg = F.normalize(torch.reshape(v_inputs, (-1, 936)))

            s_fmap_avg = s_fmap
            t_fmap_avg = torch.reshape(v_inputs, (-1, 936))

            kd = mse_loss(s_fmap_avg, t_fmap_avg)
            # kd = pair_loss(s_fmap_avg, t_fmap_avg)
            # output = student_classifier(s_fmap_avg)
            s4 = avgpool(s4)
            s4_input = s4.view(s4.size(0), -1)
            output = student_classifier(s4_input)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # output = student_classifier(s_fmap)
            # loss = criterion(output, target)
            test_loss += loss
            kd_test_loss += kd
            # npair_loss += npl
        test_loss = test_loss / len(test_loader.sampler) * test_loader.batch_size
        kd_test_loss = kd_test_loss / len(test_loader.sampler) * test_loader.batch_size
        npair_loss = npair_loss / len(test_loader.sampler) * test_loader.batch_size
        acc = 100. * correct / len(test_loader.sampler)
        format_str = 'Test set: Average loss: {:.4f}, Average Npair loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, npair_loss, correct, len(test_loader.sampler), acc)
        print(format_str)
        if to_log is not None:
            write_log(format_str, to_log)
        return test_loss.item(), acc, kd_test_loss.item()


def evaluate():
    # turn models to eval mode
    student_model.eval()

    # test
    all_pred = []
    all_target = []
    test_loss = []
    all_outputs = []

    # log dir
    logdir = os.path.join(path['dir'], 'log.txt')

    with torch.no_grad():
        for i, conc_data in enumerate(test_loader):
            h_data, v_data = conc_data
            azi, ele, target = h_data
            v_inputs, _ = v_data

            # prepare input and target
            # azi = torch.permute(azi, (0, 2, 1, 3, 4))
            # ele = torch.permute(ele, (0, 2, 1, 3, 4))
            azi = azi.to(device, dtype=torch.float)
            ele = ele.to(device, dtype=torch.float)
            # v_inputs = torch.permute(v_inputs, (0, 2, 1, 3, 4)).to(device)
            v_inputs = v_inputs.to(device)
            target = target.to(device, dtype=torch.long)

            _, g_s = student_model(azi, ele)
            s1, s2, s3, s4 = g_s
            s4 = avgpool(s4)
            s4_input = s4.view(s4.size(0), -1)
            output = student_classifier(s4_input)
            # loss
            loss = criterion_test(output, target)
            test_loss.append(loss.item())
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = pred.cpu().numpy().flatten()
            target = target.cpu().numpy().flatten()
            output = output.cpu().numpy()
            all_pred = np.concatenate((all_pred, pred), axis=0)
            all_target = np.concatenate((all_target, target), axis=0)
            all_outputs.append(output)

    all_outputs = np.concatenate(all_outputs, axis=0)
    write_log(classification_report(all_target, all_pred, target_names=emotion_list), logdir)
    return all_pred, all_target, all_outputs, test_loss




if __name__ == "__main__":

    config = dict(num_epochs=50,
                  lr=0.0006,
                  lr_step_size=20,
                  lr_decay_gamma=0.2,
                  batch_size=16,
                  num_classes=7,
                  v_num_frames=30,
                  h_num_frames=100,
                  cumulated_frame=True,
                  mse_weight=5,
                  continue_learning=True
                  )

    emotion_list = ['Neutral', 'Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    # results dir
    result_dir = "FER/results"
    path = dir_path("evaluate_Supervision_heatmap_landmark_baseline", result_dir)
    best_folder = "Supervision_heatmap_landmark_baseline_20220608-102041"
    #

    # save training config
    save_to_json(config, path['config'])

    # load data
    landmark_root = "C:/Users/Zber/Desktop/Subjects_Landmark_video"
    landmark_train_ann = os.path.join(landmark_root, 'landmark_S5_train.txt')
    landmark_test_ann = os.path.join(landmark_root, 'landmark_S5_test.txt')

    # landmark_train_ann = os.path.join(landmark_root, 'landmark_train_S5.txt')
    # landmark_test_ann = os.path.join(landmark_root, 'landmark_test_S5.txt')

    heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"
    # h_train_ann = os.path.join(heatmap_root, "heatmap_annotation_train_new.txt")
    # h_test_ann = os.path.join(heatmap_root, "heatmap_annotation_test_new.txt")

    h_train_ann = os.path.join(heatmap_root, "heatmap_train_landmark_S5.txt")
    h_test_ann = os.path.join(heatmap_root, "heatmap_test_landmark_S5.txt")

    # landmark datasets
    landmark_train = LandmarkDataset(landmark_root, landmark_train_ann, num_dim=2)
    landmark_test = LandmarkDataset(landmark_root, landmark_test_ann, num_dim=2)

    # heatmap datasets
    heatmap_train = HeatmapDataset(heatmap_root, h_train_ann, cumulated=config['cumulated_frame'],
                                   frame_cumulated=False, num_frames=config['h_num_frames'])
    heatmap_test = HeatmapDataset(heatmap_root, h_test_ann, cumulated=config['cumulated_frame'],
                                  frame_cumulated=False, num_frames=config['h_num_frames'])

    dataset_train = ConcatDataset(heatmap_train, landmark_train)
    dataset_test = ConcatDataset(heatmap_test, landmark_test)

    train_loader = DataLoader(dataset_train, num_workers=4, pin_memory=True, shuffle=True, batch_size=config['batch_size'])
    test_loader = DataLoader(dataset_test, num_workers=4, pin_memory=True, batch_size=config['batch_size'])

    # teacher model
    teacher_model = None
    # student model
    student_model = ImageSingle_v1(num_classes=config['num_classes'], block=ImageDualNet_Single_v1,
                                   subblock=ImageNet_Large_v1)

    if config['continue_learning']:
        s_checkpoint = os.path.join(result_dir, best_folder, 'model_best.pth.tar')
        student_model.load_state_dict(torch.load(s_checkpoint))
    student_model = student_model.to(device)

    student_classifier = Classifier_Transformer(input_dim=512, num_classes=config['num_classes'])
    if config['continue_learning']:
        classifier_checkpoint = os.path.join(result_dir, best_folder, 'classifier_best.pth.tar')
        student_classifier.load_state_dict(torch.load(classifier_checkpoint))
    student_classifier = student_classifier.to(device)

    avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # tf block
    tf_block = TFblock(dim_in=512, dim_inter=128, dim_out=936)
    if config['continue_learning']:
        block_checkpoint = os.path.join(result_dir, best_folder, 'block_best.pth.tar')
        tf_block.load_state_dict(torch.load(block_checkpoint))
    tf_block = tf_block.to(device)

    mse_loss = nn.MSELoss()
    criterion_test = nn.CrossEntropyLoss()

    criterions = {'criterionCls': criterion_test, 'criterionKD': mse_loss}

    optimizer = torch.optim.Adam(
        list(student_model.parameters()) + list(tf_block.parameters()) + list(student_classifier.parameters()),
        lr=config['lr'])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step_size'],
                                                   gamma=config['lr_decay_gamma'])



    metrics_dic = {
        'predict': [],
        'target': [],
    }

    pred, label, output, loss = evaluate()

    metrics_dic['predict'] = pred
    metrics_dic['target'] = label

    # save csv log
    df = pd.DataFrame.from_dict(metrics_dic)
    df.to_csv(path['metrics'], sep='\t', encoding='utf-8')

    # save output and target to numpy
    np.save(os.path.join(path['dir'], 'outputs.npy'), output)
    np.save(os.path.join(path['dir'], 'loss.npy'), np.asarray(loss))


    #
    # metrics_dic = {
    #     'train_loss': [],
    #     'train_cls_loss': [],
    #     'train_kd_loss': [],
    #     'precision': [],
    #     'test_loss': [],
    #     'test_kd_loss': [],
    # }
    #
    # best_acc = 0
    # best_loss = 100
    # for epoch in range(config['num_epochs']):
    #     loss, cls_loss, kd_loss = train(teacher_model, student_model, data_loader=train_loader, criterion=criterions,
    #                                     optimizer=optimizer, epoch=epoch, to_log=path['log'])
    #     test_loss, acc, test_kd_loss = test(student_model, teacher_model, test_loader=test_loader,
    #                                         criterion=criterion_test, to_log=path['log'])
    #     if acc >= best_acc:
    #         # if best_loss >= test_loss:
    #         best_acc = acc
    #         # best_loss = test_loss
    #         save_checkpoint(student_model.state_dict(), is_best=True, checkpoint=path['dir'], name="model")
    #         save_checkpoint(tf_block.state_dict(), is_best=True, checkpoint=path['dir'], name="block")
    #         save_checkpoint(student_classifier.state_dict(), is_best=True, checkpoint=path['dir'],
    #                         name="classifier")
    #     if (epoch + 1) % 5 == 0:
    #         save_checkpoint(student_model.state_dict(), is_best=False, checkpoint=path['dir'], epoch=epoch,
    #                         name="model")
    #         save_checkpoint(tf_block.state_dict(), is_best=False, checkpoint=path['dir'], name="block",
    #                         epoch=epoch)
    #         save_checkpoint(student_classifier.state_dict(), is_best=False, checkpoint=path['dir'],
    #                         name="classifier", epoch=epoch)
    #
    #     lr_scheduler.step()
    #     metrics_dic['train_loss'].append(loss)
    #     metrics_dic['train_cls_loss'].append(cls_loss)
    #     metrics_dic['train_kd_loss'].append(kd_loss)
    #     metrics_dic['test_loss'].append(test_loss)
    #     metrics_dic['precision'].append(acc)
    #     metrics_dic['test_kd_loss'].append(test_kd_loss)
    #
    # # print best acc after training
    # write_log("<<<<< Best Accuracy = {:.2f} >>>>>".format(best_acc), path['log'])
    #
    # # save csv log
    # df = pd.DataFrame.from_dict(metrics_dic)
    # df.to_csv(path['metrics'], sep='\t', encoding='utf-8')
    #
    # # shutdown
    # # os.system("shutdown /s /t 1")
