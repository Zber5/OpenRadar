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
    Classifier_Transformer
from models.ConvLSTM import ConvLSTMFull_ME
from metric_losses import NPairLoss_CrossModal, NPairLoss, NPairLoss_CrossModal_Weighted
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


class TFblock_v4(nn.Module):
    def __init__(self, dim_in=512, dim_inter=1024):
        super(TFblock_v4, self).__init__()
        # self.encoder = nn.Linear(dim_in, dim_inter)
        # self.decoder = nn.Linear(dim_inter, dim_in)

        self.encoder = nn.Conv2d(in_channels=dim_in, out_channels=dim_inter, kernel_size=(4, 2))
        self.bn1 = nn.BatchNorm2d(num_features=dim_inter)
        self.decoder = nn.ConvTranspose2d(in_channels=dim_inter, out_channels=(dim_in + dim_inter) // 2, kernel_size=(
            3, 3), stride=(1, 1), padding=(0, 0))
        self.decoder1 = nn.ConvTranspose2d(in_channels=(dim_in + dim_inter) // 2, out_channels=dim_in, kernel_size=(
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
        v_inputs, _ = v_data

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

        npl = npaircross(s_fmap_avg, t_fmap_avg, targets)
        # npl = npairloss(s_fmap_avg, targets)

        # s_fmap = avgpool(s_fmap)

        s_input = s_fmap.view((s_fmap.size(0), -1))
        outputs = student_classifier(s_input)

        # outputs = student_classifier(s_fmap_avg)

        cls_loss = distillation(outputs, pseudo_targets, targets, config['softmax_temperature'])
        # loss = cls_loss + 1 * npl_cross + 1 * npl + 2 * kd_loss
        # loss = cls_loss + 1 * npl_cross + 2 * kd_loss
        # loss = cls_loss + 2 * kd_loss
        # loss = cls_loss + config['npair_weight'] * npl + config['kd_weight'] * kd_loss
        loss = cls_loss + config['npair_weight'] * npl
        # loss = cls_loss + config['npair_weight'] * npl_cross + config['kd_weight'] * kd_loss

        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        cls_losses.update(cls_loss.item(), data_loader.batch_size)
        kd_losses.update(kd_loss.item(), data_loader.batch_size)
        npair_losses.update(npl.item(), data_loader.batch_size)
        npaircross_losses.update(npl.item(), data_loader.batch_size)
        # npaircross_losses.update(npl_cross.item(), data_loader.batch_size)
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


    t4_total = []
    s4_total = []
    labels = []



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

            s_fmap_avg = torch.squeeze(avgpool(s_fmap))
            t_fmap_avg = torch.squeeze(avgpool(torch.mean(t_fmap, 1)))
            s_fmap_avg = F.normalize(s_fmap_avg)
            t_fmap_avg = F.normalize(t_fmap_avg)

            t4_total.append(t_fmap_avg.cpu().numpy())
            s4_total.append(s_fmap_avg.cpu().numpy())
            labels.append(target.cpu().numpy())

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

        teacher_out = np.concatenate(t4_total, axis=0)
        student_out = np.concatenate(s4_total, axis=0)
        labels_out = np.concatenate(labels, axis=0)
        np.save(os.path.join(path['dir'], 'image'), teacher_out)
        np.save(os.path.join(path['dir'], 'kd'), student_out)
        np.save(os.path.join(path['dir'], 'labels'), labels_out)

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

    config = dict(num_epochs=1,
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
                  softmax_temperature=4.0,
                  loss_mode='cse',
                  continue_learning=True,
                  cumulated_frame=True,
                  kl_weight=1,
                  ce_weight=1,
                  npair_weight=1,
                  kd_weight=1,
                  )

    emotion_list = ['Neutral', 'Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    # results dir
    result_dir = "FER/results"
    # path = dir_path("Supervision_SUM_image2D_TransformerLarge_L3_CrossNpairLossWeighted", result_dir)
    path = dir_path("Baseline3_TSNE", result_dir)
    # path = dir_path("Supervision_SUM_image2D_TransformerLargeWithBN_L3_Cross+SelfNpairLoss_LargeClassifier", result_dir)

    # best_folder = "Supervision_image2D_Transformer_extractor_BN_L3_sum_LargeTFblock_20220303-210816"
    # best_folder = "Supervision_image2D_Transformer_extractor_BNv2_L3_sum_LargeTFblock_20220304-132937"
    # best_folder = "Supervision_SUM_image2D_TransformerLarge_L3_HeterNpairLoss_20220309-174556"
    # best_folder = "Supervision_image2D_Transformer_extractor_L3_sum_LargeTFblock_20220227-132522"
    # best_folder = "Supervision_SUM_image2D_TransformerLargeWithBN_L3_Cross+SelfNpairLoss_LargeClassifier_20220309-230901"
    best_folder = "Supervision_SUM_image2D_Baseline3_20220628-234659"

    # save training config
    save_to_json(config, path['config'])

    # New version data
    videos_root = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\'
    v_train_ann = os.path.join(videos_root, 'video_annotation_train_new.txt')
    v_test_ann = os.path.join(videos_root, 'video_annotation_test_new.txt')

    heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"
    h_train_ann = os.path.join(heatmap_root, "heatmap_annotation_train_new.txt")
    h_test_ann = os.path.join(heatmap_root, "heatmap_annotation_test_new.txt")

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
    heatmap_train = HeatmapDataset(heatmap_root, h_train_ann, cumulated=config['cumulated_frame'],
                                   frame_cumulated=False, num_frames=config['h_num_frames'])
    heatmap_test = HeatmapDataset(heatmap_root, h_test_ann, cumulated=config['cumulated_frame'],
                                  frame_cumulated=False, num_frames=config['h_num_frames'])

    dataset_train = ConcatDataset(heatmap_train, video_train)
    dataset_test = ConcatDataset(heatmap_test, video_test)

    train_loader = DataLoader(dataset_train, num_workers=2, shuffle=True, pin_memory=True,
                              batch_size=config['batch_size'])
    test_loader = DataLoader(dataset_test, num_workers=2, pin_memory=True, batch_size=config['batch_size'])

    # create model
    fmodel = resnet18()
    cmodel = Classifier(num_classes=7)
    teacher_model = ResNetFull_Teacher(fmodel, cmodel)
    avgpool = nn.AdaptiveAvgPool2d((1, 1))

    pdist = nn.PairwiseDistance()

    # teacher model
    checkpoint = os.path.join(result_dir, "Pretrained_ResNet_video_v1_20220122-002807", 'best.pth.tar')
    assert os.path.exists(checkpoint), 'Error: no checkpoint directory found!'
    teacher_model.load_state_dict(torch.load(checkpoint))
    teacher_model = teacher_model.to(device)

    # student model
    student_model = ImageSingle_v1(num_classes=config['num_classes'], block=ImageDualNet_Single_v1,
                                   subblock=ImageNet_Large_v1)

    if config['continue_learning']:
        s_checkpoint = os.path.join(result_dir, best_folder, 'model_best.pth.tar')
        student_model.load_state_dict(torch.load(s_checkpoint))
    student_model = student_model.to(device)

    student_classifier = Classifier_Transformer(input_dim=512 * 7 * 7, num_classes=config['num_classes'])
    if config['continue_learning']:
        classifier_checkpoint = os.path.join(result_dir, best_folder, 'classifier_best.pth.tar')
        student_classifier.load_state_dict(torch.load(classifier_checkpoint))
    student_classifier = student_classifier.to(device)

    # create motion extractor model
    extractor = ConvLSTMFull_ME(input_dim=1,
                                hidden_dim=[3],
                                kernel_size=(1, 1),
                                num_layers=1,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)
    # if config['continue_learning']:
    #     extractor_checkpoint = os.path.join(result_dir, best_folder, 'extractor_best.pth.tar')
    #     extractor.load_state_dict(torch.load(extractor_checkpoint))
    extractor = extractor.to(device)

    tf_block = TFblock_v4()
    if config['continue_learning']:
        block_checkpoint = os.path.join(result_dir, best_folder, 'block_best.pth.tar')
        tf_block.load_state_dict(torch.load(block_checkpoint))
    tf_block = tf_block.to(device)

    # npairloss = NPairLoss()
    npaircross = NPairLoss_CrossModal()
    # npaircross = NPairLoss_CrossModal_Weighted()

    # initialize critierion and optimizer
    # criterion = _make_criterion(alpha=config['weight_alpha'], T=config['softmax_temperature'], mode=config['loss_mode'])

    criterion_test = nn.CrossEntropyLoss()
    # criterion_cls = nn.CrossEntropyLoss()
    # criterion_kd = nn.CosineEmbeddingLoss(margin=config['loss_margin']).to(device)
    criterion_kd = None

    criterion_cls = _make_criterion(alpha=config['weight_alpha'], T=config['softmax_temperature'],
                                    mode=config['loss_mode'])

    criterions = {'criterionCls': criterion_cls, 'criterionKD': criterion_kd}

    optimizer = torch.optim.Adam(
        list(student_model.parameters()) + list(tf_block.parameters()) + list(student_classifier.parameters()) + list(
            extractor.parameters()), lr=config['lr'])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step_size'],
                                                   gamma=config['lr_decay_gamma'])

    metrics_dic = {
        'loss': [],
        'precision': [],
    }

    best_acc = 0
    best_loss = 100
    for epoch in range(config['num_epochs']):
        # train_loss = train(teacher_model, student_model, data_loader=train_loader, criterion=criterions,
        #                    optimizer=optimizer, epoch=epoch, to_log=path['log'])
        test_loss, acc = test(student_model, teacher_model, test_loader=test_loader, criterion=criterion_test,
                              to_log=path['log'])
        if acc >= best_acc:
            # if best_loss >= test_loss:
            best_acc = acc
            # best_loss = test_loss
            save_checkpoint(student_model.state_dict(), is_best=True, checkpoint=path['dir'], name="model")
            save_checkpoint(tf_block.state_dict(), is_best=True, checkpoint=path['dir'], name="block")
            save_checkpoint(student_classifier.state_dict(), is_best=True, checkpoint=path['dir'],
                            name="classifier")
            save_checkpoint(extractor.state_dict(), is_best=True, checkpoint=path['dir'],
                            name="extractor")
        if (epoch + 1) % 5 == 0:
            save_checkpoint(student_model.state_dict(), is_best=False, checkpoint=path['dir'], epoch=epoch,
                            name="model")
            save_checkpoint(tf_block.state_dict(), is_best=False, checkpoint=path['dir'], name="block",
                            epoch=epoch)
            save_checkpoint(student_classifier.state_dict(), is_best=False, checkpoint=path['dir'],
                            name="classifier", epoch=epoch)
            save_checkpoint(extractor.state_dict(), is_best=False, checkpoint=path['dir'],
                            name="extractor", epoch=epoch)

        lr_scheduler.step()

        metrics_dic['loss'].append(test_loss)
        metrics_dic['precision'].append(acc)

    # print best acc after training
    write_log("<<<<< Best Accuracy = {:.2f} >>>>>".format(best_acc), path['log'])

    # save csv log
    df = pd.DataFrame.from_dict(metrics_dic)
    df.to_csv(path['metrics'], sep='\t', encoding='utf-8')