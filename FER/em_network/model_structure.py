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
from sklearn.metrics import classification_report, confusion_matrix

from utils import accuracy, device, AverageMeter, dir_path, write_log, save_checkpoint
from models.resnet import resnet18, Classifier, ResNetFull_Teacher
from models.Conv2D import ImageSingle_Student, ImageSingle_v1, ImageDualNet_Single_v1, ImageNet_Large_v1, \
    Classifier_Transformer, ImageNet_Small_v1
from models.ConvLSTM import ConvLSTMFull_ME
from metric_losses import NPairLoss_CrossModal, NPairLoss, NPairLoss_CrossModal_Weighted
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchsummary import summary
# torch.autograd.set_detect_anomaly(True)

import os

os.chdir(ROOT_PATH)

# set seed, make result reporducable
SEED = 1111
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


# class NpairLoss(nn.Module):
#     """the multi-class n-pair loss"""
#
#     def __init__(self, l2_reg=0.02):
#         super(NpairLoss, self).__init__()
#         self.l2_reg = l2_reg
#
#     def forward(self, anchor, positive, target):
#         batch_size = anchor.size(0)
#         target = target.view(target.size(0), 1)
#
#         target = (target == torch.transpose(target, 0, 1)).float()
#         target = target / torch.sum(target, dim=1, keepdim=True).float()
#
#         logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
#         loss_ce = cross_entropy(logit, target)
#         l2_loss = torch.sum(anchor ** 2) / batch_size + torch.sum(positive ** 2) / batch_size
#
#         loss = loss_ce + self.l2_reg * l2_loss * 0.25
#         return loss


class NST(nn.Module):
    '''
    Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
    https://arxiv.org/pdf/1707.01219.pdf
    '''

    def __init__(self):
        super(NST, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), fm_s.size(1), -1)
        fm_s = F.normalize(fm_s, dim=2)

        fm_t = fm_t.view(fm_t.size(0), fm_t.size(1), -1)
        fm_t = F.normalize(fm_t, dim=2)

        loss = self.poly_kernel(fm_t, fm_t).mean() \
               + self.poly_kernel(fm_s, fm_s).mean() \
               - 2 * self.poly_kernel(fm_s, fm_t).mean()

        return loss

    def poly_kernel(self, fm1, fm2):
        fm1 = fm1.unsqueeze(1)
        fm2 = fm2.unsqueeze(2)
        out = (fm1 * fm2).sum(-1).pow(2)

        return out


class SP(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''

    def __init__(self):
        super(SP, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)

        fm_t = fm_t.view(fm_t.size(0), -1)
        G_t = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)

        loss = F.mse_loss(norm_G_s, norm_G_t)

        return loss


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


class TFblock(nn.Module):
    def __init__(self, dim_in=521, dim_inter=128):
        super(TFblock, self).__init__()
        # self.encoder = nn.Linear(dim_in, dim_inter)
        # self.decoder = nn.Linear(dim_inter, dim_in)

        self.encoder = nn.Conv2d(dim_in, dim_inter, 1)
        self.decoder = nn.Conv2d(dim_inter, dim_in, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class TFblock_v2(nn.Module):
    def __init__(self, dim_in=521, dim_inter=128):
        super(TFblock_v2, self).__init__()
        # self.encoder = nn.Linear(dim_in, dim_inter)
        # self.decoder = nn.Linear(dim_inter, dim_in)

        self.encoder = nn.Conv2d(dim_in, dim_inter, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Conv2d(dim_inter, dim_in, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        x = self.decoder(x)

        return x


class TFblock_v3(nn.Module):
    def __init__(self, dim_in=512, dim_inter=1024):
        super(TFblock_v3, self).__init__()
        # self.encoder = nn.Linear(dim_in, dim_inter)
        # self.decoder = nn.Linear(dim_inter, dim_in)

        self.encoder = nn.Conv2d(in_channels=dim_in, out_channels=dim_inter, kernel_size=(4, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=dim_inter)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.decoder = nn.ConvTranspose2d(in_channels=dim_inter, out_channels=(dim_in + dim_inter) // 2, kernel_size=(
            3, 3), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=(dim_in + dim_inter) // 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.decoder1 = nn.ConvTranspose2d(in_channels=(dim_in + dim_inter) // 2, out_channels=dim_in, kernel_size=(
            5, 5), stride=(1, 1), padding=(0, 0), bias=False)
        self.sg = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        # x = self.avgpool(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.decoder(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.decoder1(x)
        x = self.sg(x)

        return x


class TFblock_v4(nn.Module):
    def __init__(self, dim_in=512, dim_inter=1024):
        super(TFblock_v4, self).__init__()
        # self.encoder = nn.Linear(dim_in, dim_inter)
        # self.decoder = nn.Linear(dim_inter, dim_in)

        self.encoder = nn.Conv2d(in_channels=dim_in, out_channels=dim_inter, kernel_size=(4, 2))
        self.bn1 = nn.BatchNorm2d(num_features=dim_inter)
        self.decoder = nn.ConvTranspose2d(in_channels=dim_inter, out_channels=(dim_in + dim_inter) // 2, kernel_size=(
            3, 3), stride=(1, 1), padding=(0, 0))
        self.decoder1 = nn.ConvTranspose2d(in_channels=(dim_in + dim_inter) // 2, out_channels=dim_in * 2, kernel_size=(
            5, 5), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        x = self.encoder(x)
        # x = self.avgpool(x)
        x = self.bn1(x)
        x = self.decoder(x)
        x = self.decoder1(x)
        return x


def distillation(y, teacher_scores, labels, T, alpha=0.5):
    # p = F.log_softmax(y, dim=1)
    p = F.log_softmax(y / T, dim=1)
    # q = F.softmax(teacher_scores, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T ** 2) / y.shape[0]
    # l_kl = F.kl_div(p, q, size_average=False)
    l_ce = F.cross_entropy(y, labels)
    # return l_kl * alpha + l_ce * (1. - alpha)
    return config['kl_weight'] * l_kl + config['ce_weight'] * l_ce
    # return config['ce_weight'] * l_ce


def at(x):
    if len(x.shape) != 5:
        return F.normalize(x.pow(2).mean((2, 3)).view(x.size(0), -1))
    else:
        return F.normalize(x.pow(2).mean((1, 3, 4)).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


def pair_loss(x, y):
    x = at(x)
    y = at(y)
    diff = pdist(x, y).mean()
    return diff


def pair_loss_old(x, y):
    x = at(x)
    y = at(y)
    diff = pdist(x, y).pow(2).mean()
    return diff


def at_v1(x):
    if len(x.shape) != 5:
        return F.normalize(x.pow(2).view(x.size(0), -1))
    else:
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss_v1(x, y):
    return (at_v1(x) - at_v1(y)).pow(2).mean()


def pair_loss_old_v1(x, y):
    x = at_v1(x)
    y = at_v1(y)
    diff = pdist(x, y).pow(2).mean()
    return diff


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
    teacher_model.eval()
    student_model.train()
    tf_block.train()
    student_classifier.train()
    extractor.train()

    # record start time
    start = time.time()

    criterion_cls = criterions['criterionCls']
    criterion_kd = criterions['criterionKD']

    for i, conc_data in enumerate(data_loader):
        h_data, v_data = conc_data
        azi, ele, targets = h_data
        v_inputs, v_targets = v_data

        # prepare input and target
        # azi = torch.permute(azi, (0, 2, 1, 3, 4))
        # ele = torch.permute(ele, (0, 2, 1, 3, 4))
        azi = azi.to(device, dtype=torch.float)
        ele = ele.to(device, dtype=torch.float)
        # v_inputs = torch.permute(v_inputs, (0, 2, 1, 3, 4)).to(device)
        v_inputs = v_inputs.to(device)
        targets = targets.to(device, dtype=torch.long)

        # measure data loading time
        data_time.update(time.time() - start)

        # zero the parameter gradients
        optimizer.zero_grad()

        # gradient and do SGD step
        pseudo_targets, g_t = teacher_model(v_inputs)
        # azi, ele = extractor(azi, ele)
        _, g_s = student_model(azi, ele)

        t1, t2, t3, t4 = g_t
        s1, s2, s3, s4 = g_s

        t_fmap = t4
        s_fmap = s4

        s_fmap = tf_block(s_fmap)
        kd_loss = pair_loss_old_v1(s_fmap, t_fmap)
        # kd_loss = at_loss_v1(s_fmap, t_fmap)

        s_fmap_avg = torch.squeeze(avgpool(s_fmap))
        t_fmap_avg = torch.squeeze(avgpool(torch.mean(t_fmap, 1)))
        s_fmap_avg = F.normalize(s_fmap_avg)
        t_fmap_avg = F.normalize(t_fmap_avg)

        # npl_cross = npaircross(s_fmap_avg, t_fmap_avg, targets)
        npl = npairloss(s_fmap_avg, targets)

        # s_fmap = avgpool(s_fmap)

        s_input = s_fmap.view((s_fmap.size(0), -1))
        outputs = student_classifier(s_input)

        # outputs = student_classifier(s_fmap_avg)

        cls_loss = distillation(outputs, pseudo_targets, targets, config['softmax_temperature'])
        # loss = cls_loss + 1 * npl_cross + 1 * npl + 2 * kd_loss
        # loss = cls_loss + 1 * npl_cross + 2 * kd_loss
        # loss = cls_loss + 2 * kd_loss
        # if npl_cross.item() == 0:
        #     loss = cls_loss + config['kd_weight'] * kd_loss
        # else:
        # loss = cls_loss + config['npair_weight'] * npl_cross + config['kd_weight'] * kd_loss
        loss = cls_loss + config['npair_weight'] * npl + config['kd_weight'] * kd_loss

        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        cls_losses.update(cls_loss.item(), data_loader.batch_size)
        kd_losses.update(kd_loss.item(), data_loader.batch_size)
        npair_losses.update(npl.item(), data_loader.batch_size)
        npaircross_losses.update(npl.item(), data_loader.batch_size)
        # npair_losses.update(0, data_loader.batch_size)
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
                   'NPairLoss:{npair_losses.val:.4f} ({npair_losses.avg:.4f})  '
                   'NPairCrossLoss:{npaircross_losses.val:.4f} ({npair_losses.avg:.4f})  '
                   'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1: {top1.val:3.3f} ({top1.avg:3.3f})\t'
                   'Prec@5: {top5.val:3.3f} ({top5.avg:3.3f})'.format(
                epoch, i, len(data_loader), batch_time=batch_time, cls_losses=cls_losses, kd_losses=kd_losses,
                npair_losses=npair_losses, npaircross_losses=npaircross_losses, data_time=data_time, loss=losses,
                top1=top1, top5=top5))
            print(str)

            if to_log is not None:
                write_log(str + '\n', to_log)

    return train_loss


def test(s_model, t_model, test_loader, criterion, to_log=None):
    s_model.eval()
    t_model.eval()
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

            # gradient and do SGD step
            pseudo_targets, g_t = t_model(v_inputs)

            # appley motion extractor before feed into student model
            # azi, ele = extractor(azi, ele)

            _, g_s = student_model(azi, ele)

            t1, t2, t3, t4 = g_t
            s1, s2, s3, s4 = g_s

            t_fmap = t4
            s_fmap = s4
            s_fmap = tf_block(s_fmap)
            kd_loss = pair_loss_old_v1(s_fmap, t_fmap)
            # s_fmap = avgpool(s_fmap)

            # s_fmap_avg = torch.squeeze(avgpool(s_fmap))
            # t_fmap_avg = torch.squeeze(avgpool(torch.mean(t_fmap, 1)))
            # s_fmap_avg = F.normalize(s_fmap_avg)
            # t_fmap_avg = F.normalize(t_fmap_avg)

            # npl = npairloss(s_fmap_avg, t_fmap_avg, targets)
            # npl = npaircross(s_fmap_avg, t_fmap_avg, targets)
            # npl = npairloss(s_fmap_avg, target)

            s_input = s_fmap.view((s_fmap.size(0), -1))
            output = student_classifier(s_input)

            # output = student_classifier(s_fmap_avg)

            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # output = student_classifier(s_fmap)
            # loss = criterion(output, target)
            test_loss += loss
            kd_test_loss += kd_loss
            # npair_loss += npl
        test_loss = test_loss / len(test_loader.sampler) * test_loader.batch_size
        kd_loss = kd_loss / len(test_loader.sampler) * test_loader.batch_size
        npair_loss = npair_loss / len(test_loader.sampler) * test_loader.batch_size
        acc = 100. * correct / len(test_loader.sampler)
        format_str = 'Test set: Average loss: {:.4f}, Average KD loss: {:.4f}, Average Npair loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, kd_test_loss, npair_loss, correct, len(test_loader.sampler), acc)
        print(format_str)
        if to_log is not None:
            write_log(format_str, to_log)
        return test_loss.item(), acc


def evaluate(student_model, resdir, testloader):
    # load weights
    cmdir = os.path.join(resdir, 'cm.pdf')
    logdir = os.path.join(resdir, 'cm_log.txt')
    model_path = os.path.join(resdir, 'best.pth.tar')

    # load weights

    student_model.eval()

    # test
    all_pred = []
    all_target = []
    test_loss = 0
    with torch.no_grad():
        for i, conc_data in enumerate(test_loader):
            # prepare input and target to device
            h_data, v_data = conc_data
            azi, ele, target = h_data
            v_inputs, _ = v_data

            _, g_s = student_model(azi, ele)

            s1, s2, s3, s4 = g_s

            s_fmap = s4
            s_fmap = tf_block(s_fmap)

            s_input = s_fmap.view((s_fmap.size(0), -1))
            output = student_classifier(s_input)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = pred.cpu().numpy().flatten()
            target = target.cpu().numpy().flatten()
            all_pred = np.concatenate((all_pred, pred), axis=0)
            all_target = np.concatenate((all_target, target), axis=0)
    # print
    write_log(classification_report(all_target, all_pred, target_names=emotion_list), logdir)

    cm = confusion_matrix(all_target, all_pred)

    ax = sns.heatmap(cm, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Classes')
    ax.set_ylabel('Actual Classes')
    ax.xaxis.set_ticklabels(emotion_list)
    ax.yaxis.set_ticklabels(emotion_list)
    plt.savefig(cmdir)


if __name__ == "__main__":

    config = dict(num_epochs=40,
                  lr=0.0006,
                  lr_step_size=20,
                  lr_decay_gamma=0.2,
                  batch_size=16,
                  num_classes=7,
                  v_num_frames=30,
                  h_num_frames=100,
                  imag_size=224,
                  # weight_alpha=0.7,
                  # softmax_temperature=16.0,
                  # loss_mode='cse'
                  lambda_kd=1000,
                  loss_margin=0.1,
                  weight_alpha=0.7,
                  softmax_temperature=8.0,
                  loss_mode='cse',
                  continue_learning=False,
                  cumulated_frame=True,
                  kl_weight=1,
                  ce_weight=1,
                  npair_weight=1,
                  kd_weight=1,
                  )

    emotion_list = ['Neutral', 'Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    fmodel = resnet18()
    cmodel = Classifier(num_classes=7)
    teacher_model = ResNetFull_Teacher(fmodel, cmodel)
    teacher_model = teacher_model.to(device)

    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    pdist = nn.PairwiseDistance()

    # teacher model



    # student model
    student_model = ImageSingle_v1(num_classes=config['num_classes'], block=ImageDualNet_Single_v1,
                                   subblock=ImageNet_Small_v1)
    student_model = student_model.to(device)

    student_classifier = Classifier_Transformer(input_dim=512 * 7 * 7, num_classes=config['num_classes'])
    student_classifier = student_classifier.to(device)

    # create motion extractor model
    extractor = ConvLSTMFull_ME(input_dim=1,
                                hidden_dim=[3],
                                kernel_size=(1, 1),
                                num_layers=1,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)


    extractor = extractor.to(device)

    # tf block
    # tf_block = TFblock(dim_in=512, dim_inter=128)
    tf_block = TFblock_v4(dim_in=256)

    tf_block = tf_block.to(device)

    npairloss = NPairLoss()
    # npaircross = NPairLoss_CrossModal()
    npaircross = NPairLoss_CrossModal_Weighted()

    # initialize critierion and optimizer
    # criterion = _make_criterion(alpha=config['weight_alpha'], T=config['softmax_temperature'], mode=config['loss_mode'])

    criterion_test = nn.CrossEntropyLoss()
    # criterion_cls = nn.CrossEntropyLoss()
    # criterion_kd = nn.CosineEmbeddingLoss(margin=config['loss_margin']).to(device)

    # model = model.to(device)
    # input1 = torch.randn(3, 224, 224).to(device)
    model = ImageNet_Small_v1().to(device)
    summary(model, (1, 30, 5))

    print("")
    # summary(model, (12, 10, 100))

    # input1 = torch.randn(8, 1, 91, 10)
    # input1 = torch.randn(8, 1, 70, 70)
    # input1 = input1.to(device)

    # input2 = torch.randn(8, 1, 91, 10)
    # input2 = torch.randn(8, 1, 70, 70)
    # input2 = input2.to(device)

    # input3 = torch.randn(8, 12, 10, 100)
    # input3 = torch.permute(input3, (0, 2, 3, 1))
    # input3 = torch.reshape(input3, (input3.size(0), -1, input3.size(3)))
    # input3 = input3.to(device)
    # out = model(input1)
    # azi, ele = model(input1, input2)
    # out, g = model(input1, input2)
    # out = model(input1, input2, input3)
    # print(out.size())