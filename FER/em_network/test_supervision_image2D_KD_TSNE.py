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
from models.resnet import resnet18, Classifier, ResNetFull_Teacher
from models.Conv2D import ImageSingle_Student, ImageSingle_v1, ImageDualNet_Single_v1, ImageNet_Large_v1, \
    Classifier_Transformer
import pandas as pd
from metric_losses import NPairLoss_CrossModal, NPairLoss, NPairLoss_CrossModal_Weighted, TupleSampler
import os

os.chdir(ROOT_PATH)

# set seed, make result reporducable
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


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


def pair_loss_old_v1(x, y):
    x = at_v1(x)
    y = at_v1(y)
    diff = pdist(x, y).pow(2).mean()
    return diff


def pair_distance(x, y):
    pdist = nn.PairwiseDistance()
    diff = pdist(x, y).mean()
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

    epoch_kd_losses = []
    epoch_npl_losses = []
    epoch_cls_losses = []
    epoch_dist_losses = []

    # switch to train mode
    teacher_model.eval()
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
        _, g_s = student_model(azi, ele)

        t1, t2, t3, t4 = g_t
        s1, s2, s3, s4 = g_s
        t_fmap = t4
        s_fmap = s4

        s_fmap = tf_block(s_fmap)
        dis_loss = pair_loss_old_v1(s_fmap, t_fmap)

        s_fmap_avg = torch.squeeze(avgpool(s_fmap))
        t_fmap_avg = torch.squeeze(avgpool(torch.mean(t_fmap, 1)))
        s_fmap_avg = F.normalize(s_fmap_avg)
        t_fmap_avg = F.normalize(t_fmap_avg)

        npl_loss = npairloss(s_fmap_avg, targets)
        # npl_loss = npaircross(s_fmap_avg, targets)

        s_input = s_fmap.view((s_fmap.size(0), -1))
        outputs = student_classifier(s_input)

        cls_loss = criterion_test(outputs, targets)
        kd_loss = distillation(outputs, pseudo_targets, targets, config['softmax_temperature'])

        kl = kl_loss(outputs, pseudo_targets)

        # supervised loss
        # loss = kl + dis_loss

        # kd lss
        # loss = kd_loss

        # ours
        loss = kl + cls_loss + npl_loss + dis_loss

        loss.backward()
        optimizer.step()

        epoch_kd_losses.append(kd_loss.item())
        epoch_npl_losses.append(npl_loss.item())
        epoch_cls_losses.append(cls_loss.item())
        epoch_dist_losses.append(dis_loss.item())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        cls_losses.update(cls_loss.item(), data_loader.batch_size)
        kd_losses.update(kd_loss.item(), data_loader.batch_size)
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
                   'Cls:{cls_losses.val:.4f} ({cls_losses.avg:.4f})\t'
                   'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1: {top1.val:3.3f} ({top1.avg:3.3f})\t'
                   'Prec@5: {top5.val:3.3f} ({top5.avg:3.3f})'.format(
                epoch, i, len(data_loader), batch_time=batch_time, cls_losses=cls_losses,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            print(str)

            if to_log is not None:
                write_log(str + '\n', to_log)

    return np.mean(epoch_cls_losses), np.mean(epoch_npl_losses), np.mean(epoch_kd_losses), np.mean(epoch_dist_losses)


def test(model, test_loader, criterion, to_log=None):
    teacher_model.eval()
    student_model.eval()
    tf_block.eval()
    student_classifier.eval()
    test_loss = 0
    correct = 0

    test_kd_losses = []
    test_npl_losses = []
    test_dist_losses = []
    test_cls_losses = []

    total_distance_pdist = []
    total_cross_negative_distance = []
    total_self_negative_distance = []

    sampler = TupleSampler('npair')

    t4_total = []
    s4_total = []
    labels = []

    with torch.no_grad():
        for i, conc_data in enumerate(test_loader):
            h_data, v_data = conc_data
            azi, ele, targets = h_data
            v_inputs, _ = v_data

            # prepare input and target
            azi = azi.to(device, dtype=torch.float)
            ele = ele.to(device, dtype=torch.float)
            v_inputs = v_inputs.to(device)
            targets = targets.to(device, dtype=torch.long)

            pseudo_targets, g_t = teacher_model(v_inputs)
            outputs, g_s = student_model(azi, ele)

            t1, t2, t3, t4 = g_t
            s1, s2, s3, s4 = g_s

            # kd tsne
            t4_avg = F.normalize(avgpool(torch.mean(t4, 1)))
            s4_avg = F.normalize(s4.view((s4.size(0), -1)))

            t4_total.append(torch.squeeze(t4_avg).cpu().numpy())
            s4_total.append(torch.squeeze(s4_avg).cpu().numpy())
            labels.append(targets.cpu().numpy())

            t_fmap = t4
            s_fmap = s4

            s_fmap = tf_block(s_fmap)
            dis_loss = pair_loss_old_v1(s_fmap, t_fmap)

            s_fmap_avg = torch.squeeze(avgpool(s_fmap))
            t_fmap_avg = torch.squeeze(avgpool(torch.mean(t_fmap, 1)))
            s_fmap_avg = F.normalize(s_fmap_avg)

            npl_loss = npairloss(s_fmap_avg, targets)

            s_input = s_fmap.view((s_fmap.size(0), -1))
            outputs = student_classifier(s_input)

            cls_loss = criterion_test(outputs, targets)
            kd_loss = distillation(outputs, pseudo_targets, targets, config['softmax_temperature'])

            t_fmap = t4
            s_fmap = tf_block(s4)
            s_fmap = at(s_fmap)
            t_fmap = at(t_fmap)

            sampled_npairs = sampler.give(s_fmap, targets)

            for npair in sampled_npairs:
                anchor = s_fmap[npair[0]:npair[0] + 1, :]
                cross_negatives = t_fmap[npair[2:], :]
                self_negatives = s_fmap[npair[2:], :]

                anchors = torch.cat([anchor for i in range(cross_negatives.size(0))], dim=0)
                cross_nega_dist = pair_distance(anchors, cross_negatives)
                self_nega_dist = pair_distance(anchors, self_negatives)
                total_cross_negative_distance.append(float(cross_nega_dist))
                total_self_negative_distance.append(float(self_nega_dist))

            dists = pair_distance(s_fmap, t_fmap)
            total_distance_pdist.append(float(dists))

            test_loss += cls_loss
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

            test_kd_losses.append(kd_loss.item())
            test_dist_losses.append(dis_loss.item())
            test_npl_losses.append(npl_loss.item())
            test_cls_losses.append(cls_loss.item())

        # t4, s4
        teacher_out = np.concatenate(t4_total, axis=0)
        student_out = np.concatenate(s4_total, axis=0)
        labels_out = np.concatenate(labels, axis=0)
        np.save(os.path.join(path['dir'], 'image'), teacher_out)
        np.save(os.path.join(path['dir'], 'kd'), student_out)
        np.save(os.path.join(path['dir'], 'labels'), labels_out)

        test_kd_loss = np.mean(test_kd_losses)
        test_dist_loss = np.mean(test_dist_losses)
        test_npl_loss = np.mean(test_npl_losses)
        test_cls_loss = np.mean(test_cls_losses)

        test_pdist = np.mean(total_distance_pdist)
        test_cpdist = np.mean(total_cross_negative_distance)
        test_spdist = np.mean(total_self_negative_distance)

        test_loss /= len(test_loader.sampler)
        test_loss *= test_loader.batch_size
        acc = 100. * correct / len(test_loader.sampler)
        format_str = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.sampler), acc
        )
        print(format_str)
        if to_log is not None:
            write_log(format_str, to_log)
        return test_cls_loss, test_kd_loss, test_dist_loss, test_npl_loss, test_pdist, test_cpdist, \
               test_spdist, acc


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


def distillation(y, teacher_scores, labels, T, alpha=0.5):
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T ** 2) / y.shape[0]
    # l_kl = F.kl_div(p, q, size_average=False)
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)


def at(x):
    if len(x.shape) != 5:
        return F.normalize(x.pow(2).mean((2, 3)).view(x.size(0), -1))
    else:
        return F.normalize(x.pow(2).mean((1, 3, 4)).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


def at_v1(x):
    if len(x.shape) != 5:
        return F.normalize(x.pow(2).view(x.size(0), -1))
    else:
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def kl_loss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)
    return l_kl


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
                  softmax_temperature=8.0,
                  loss_mode='cse'
                  )

    # results dir
    result_dir = "FER/results"
    path = dir_path("KD_TSNE", result_dir)

    # KD best model
    # best_folder = "Supervision_SUM_image2D_KD_baseline_20220608-223236"
    best_folder = "Supervision_image2D_KD_20220204-234440"


    # save training config
    save_to_json(config, path['config'])

    # load data
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
    heatmap_train = HeatmapDataset(heatmap_root, h_train_ann, cumulated=True, num_frames=config['h_num_frames'])
    heatmap_test = HeatmapDataset(heatmap_root, h_test_ann, cumulated=True, num_frames=config['h_num_frames'])

    dataset_train = ConcatDataset(heatmap_train, video_train)
    dataset_test = ConcatDataset(heatmap_test, video_test)

    train_loader = DataLoader(dataset_train, num_workers=2, pin_memory=True, batch_size=config['batch_size'])
    test_loader = DataLoader(dataset_test, num_workers=2, pin_memory=True, batch_size=config['batch_size'])

    # create model
    fmodel = resnet18()
    cmodel = Classifier(num_classes=7)
    teacher_model = ResNetFull_Teacher(fmodel, cmodel)

    # teacher model
    checkpoint = os.path.join(result_dir, "Pretrained_ResNet_video_v1_20220122-002807", 'best.pth.tar')
    assert os.path.exists(checkpoint), 'Error: no checkpoint directory found!'
    teacher_model.load_state_dict(torch.load(checkpoint))
    teacher_model = teacher_model.to(device)

    # student model
    student_model = ImageSingle_v1(num_classes=config['num_classes'], block=ImageDualNet_Single_v1,
                                   subblock=ImageNet_Large_v1)

    # s_checkpoint = os.path.join(result_dir, best_folder, 'model_best.pth.tar')
    s_checkpoint = os.path.join(result_dir, best_folder, 'best.pth.tar')
    student_model.load_state_dict(torch.load(s_checkpoint))

    student_model = student_model.to(device)

    # tf block
    tf_block = TFblock_v4()
    # block_checkpoint = os.path.join(result_dir, best_folder, 'block_best.pth.tar')
    # # block_checkpoint = os.path.join(result_dir, best_folder, 'block_9.pth.tar')
    # tf_block.load_state_dict(torch.load(block_checkpoint))
    tf_block = tf_block.to(device)

    # student classifier
    student_classifier = Classifier_Transformer(input_dim=512 * 7 * 7, num_classes=config['num_classes'])
    # classifier_checkpoint = os.path.join(result_dir, best_folder, 'classifier_best.pth.tar')
    # student_classifier.load_state_dict(torch.load(classifier_checkpoint))
    student_classifier = student_classifier.to(device)

    # initialize critierion and optimizer
    # criterion = _make_criterion(alpha=config['weight_alpha'], T=config['softmax_temperature'], mode=config['loss_mode'])

    criterion_test = nn.CrossEntropyLoss()
    pdist = nn.PairwiseDistance()
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    npairloss = NPairLoss()
    npaircross = NPairLoss_CrossModal_Weighted()

    # criterion_cls = nn.CrossEntropyLoss()
    # criterion_kd = nn.CosineEmbeddingLoss(margin=config['loss_margin']).to(device)
    criterion_kd = NST().to(device)

    criterion_cls = _make_criterion(alpha=config['weight_alpha'], T=config['softmax_temperature'],
                                    mode=config['loss_mode'])

    criterions = {'criterionCls': criterion_cls, 'criterionKD': criterion_kd}

    optimizer = torch.optim.Adam(
        list(student_model.parameters()) + list(tf_block.parameters()) + list(student_classifier.parameters()),
        lr=config['lr'])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step_size'],
                                                   gamma=config['lr_decay_gamma'])

    metrics_dic = {
        'train_cls_loss': [],
        'train_npl_loss': [],
        'train_kd_loss': [],
        'train_dist_loss': [],
        'test_cls_loss': [],
        'test_kd_loss': [],
        'test_dist_loss': [],
        'test_npl_loss': [],
        'test_pdist': [],
        'test_cpdist': [],
        'test_spdist': [],
        'test_acc': [],
    }

    best_acc = 0
    for epoch in range(config['num_epochs']):
        # train_losses = train(teacher_model, student_model, data_loader=train_loader, criterion=criterions,
        #                      optimizer=optimizer, epoch=epoch, to_log=path['log'])
        # train_cls_loss, train_npl_loss, train_kd_loss, train_dist_loss = train_losses
        test_losses = test(student_model, test_loader=test_loader, criterion=criterion_test, to_log=path['log'])

        test_cls_loss, test_kd_loss, test_dist_loss, test_npl_loss, test_pdist, test_cpdist, \
        test_spdist, acc = test_losses
        if acc >= best_acc:
            # if best_loss >= test_loss:
            best_acc = acc
            # best_loss = test_loss
            save_checkpoint(student_model.state_dict(), is_best=True, checkpoint=path['dir'], name="model")
        elif (epoch + 1) % 5 == 0:
            save_checkpoint(student_model.state_dict(), is_best=False, checkpoint=path['dir'], epoch=epoch,
                            name="model")

        lr_scheduler.step()

        # metrics_dic['train_cls_loss'].append(train_cls_loss)
        # metrics_dic['train_npl_loss'].append(train_npl_loss)
        # metrics_dic['train_kd_loss'].append(train_kd_loss)
        # metrics_dic['train_dist_loss'].append(train_dist_loss)
        # metrics_dic['test_cls_loss'].append(test_cls_loss)
        # metrics_dic['test_kd_loss'].append(test_kd_loss)
        # metrics_dic['test_dist_loss'].append(test_dist_loss)
        # metrics_dic['test_npl_loss'].append(test_npl_loss)
        # metrics_dic['test_pdist'].append(test_pdist)
        # metrics_dic['test_cpdist'].append(test_cpdist)
        # metrics_dic['test_spdist'].append(test_spdist)
        # metrics_dic['test_acc'].append(acc)

    # print best acc after training
    write_log("<<<<< Best Accuracy = {:.2f} >>>>>".format(best_acc), path['log'])

    # save csv log
    df = pd.DataFrame.from_dict(metrics_dic)
    df.to_csv(path['metrics'], sep='\t', encoding='utf-8')

