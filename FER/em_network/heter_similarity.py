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
from metric_losses import TupleSampler

from scipy.stats import wasserstein_distance

import os

os.chdir(ROOT_PATH)

# set seed, make result reporducable
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


def get_similarity(tx, ty):
    # not completed yet
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    pdist = nn.PairwiseDistance()
    return 0


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


def load_model(model, checkpoints):
    assert os.path.exists(checkpoints), 'Error: no checkpoint directory found!'
    model.load_state_dict(torch.load(checkpoints))
    loaded_model = model.to(device)
    return loaded_model


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


def at(x):
    if len(x.shape) != 5:
        return F.normalize(x.pow(2).mean((2, 3)).view(x.size(0), -1))
    else:
        return F.normalize(x.pow(2).mean((1, 3, 4)).view(x.size(0), -1))


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


def pair_loss(x, y):
    pdist = nn.PairwiseDistance()
    x = at(x)
    y = at(y)
    # diff = pdist(x, y).pow(2).mean()
    diff = pdist(x, y).mean()
    return diff


def pair_distance(x, y):
    pdist = nn.PairwiseDistance()
    diff = pdist(x, y).mean()
    return diff


def test_metric(test_loader, student_model, teacher_model):
    student_model.eval()
    teacher_model.eval()
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    pdist = nn.PairwiseDistance()
    sampler = TupleSampler('npair')

    total_distance_mmd = []
    total_distance_mmd_rbf = []
    total_distance_pdist = []
    total_distance_wad = []
    total_distance = []
    total_cross_negative_distance = []
    total_self_negative_distance = []

    # get data
    for i, conc_data in enumerate(test_loader):
        h_data, v_data = conc_data
        azi, ele, targets = h_data
        v_inputs, _ = v_data

        # prepare input and target
        azi = azi.to(device, dtype=torch.float)
        ele = ele.to(device, dtype=torch.float)
        # v_inputs = torch.permute(v_inputs, (0, 2, 1, 3, 4)).to(device)
        v_inputs = v_inputs.to(device)
        targets = targets.to(device, dtype=torch.long)

        pseudo_targets, g_t = teacher_model(v_inputs)
        outputs, g_s = student_model(azi, ele)

        t1, t2, t3, t4 = g_t
        s1, s2, s3, s4 = g_s

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
    test_pdist = np.mean(total_distance_pdist)
    test_cpdist = np.mean(total_distance_pdist)
    test_spdist = np.mean(total_self_negative_distance)

    print("Pairwise Distance: {:0.4f}".format(np.mean(total_distance_pdist)))
    print("Cross Negative Pairwise Distance: {:0.4f}".format(np.mean(total_cross_negative_distance)))
    print("Self Negative Pairwise Distance: {:0.4f}".format(np.mean(total_self_negative_distance)))
    return test_pdist, test_cpdist, test_spdist


# def normalise_at(x):
#     s_fmap_avg = torch.squeeze(avgpool(x))
#     t_fmap_avg = torch.squeeze(avgpool(torch.mean(t_fmap, 1)))
#     s_fmap_avg = F.normalize(s_fmap_avg)
#     t_fmap_avg = F.normalize(t_fmap_avg)


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
                  loss_mode='cse'
                  )

    # KD
    # mode = "kd_baseline"
    # best_folder = "Supervision_SUM_image2D_KD_baseline_20220310-130457"
    # best_folder = "Supervision_image2D_Transformer_power_20220214-155040"

    # AE
    # mode = "ae_baseline"
    # unsupervised
    # best_folder = "Supervision_SUM_image2D_TransformerLarge_L3_Baseline3_20220310-114042"

    # cross
    # mode = "cross_npair"
    # best_folder = "Supervision_SUM_image2D_TransformerLarge_L3_HeterNpairLoss_20220309-174556"
    # best_folder = "Supervision_SUM_image2D_TransformerLarge_L3_CrossNpairLossWeighted_20220311-182907"

    # self
    # mode = "self_npair"
    # best_folder = "Supervision_SUM_image2D_TransformerLarge_L3_NpairLoss_20220309-165846"

    # self+cross
    mode = "cross+self_npair"
    best_folder = "Supervision_SUM_image2D_TransformerLargeWithBN_L3_Cross+SelfNpairLoss_LargeClassifier_20220309-230901"

    # results dir
    # result_dir = "FER/results"
    result_dir = "D:\\mmWave_FER_results\\results"

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

    train_loader = DataLoader(dataset_train, num_workers=4, pin_memory=True, batch_size=config['batch_size'])
    test_loader = DataLoader(dataset_test, num_workers=4, pin_memory=True, batch_size=config['batch_size'])

    # load teacher model
    fmodel = resnet18()
    cmodel = Classifier(num_classes=7)
    teacher_model = ResNetFull_Teacher(fmodel, cmodel)
    checkpoint = os.path.join(result_dir, "Pretrained_ResNet_video_v1_20220122-002807", 'best.pth.tar')
    teacher_model = load_model(teacher_model, checkpoint)

    # student model
    student_model = ImageSingle_v1(num_classes=config['num_classes'], block=ImageDualNet_Single_v1,
                                   subblock=ImageNet_Large_v1)
    s_checkpoint = os.path.join(result_dir, best_folder, 'model_best.pth.tar')
    student_model = load_model(student_model, s_checkpoint)

    # tf block
    if mode == "kd_baseline":
        tf_block = TFblock_v2(dim_in=512, dim_inter=128)
    else:
        tf_block = TFblock_v4(dim_in=512, dim_inter=1024)
    # tf_block = TFblock(dim_in=512, dim_inter=128)
    # if mode != 'kd_baseline':
    block_checkpoint = os.path.join(result_dir, best_folder, 'block_best.pth.tar')
    tf_block = load_model(tf_block, block_checkpoint)

    # student classifier
    # student_classifier = Classifier_Transformer(input_dim=512 * 7 * 7, num_classes=config['num_classes'])
    # if mode != 'kd_baseline':
    #     classifier_checkpoint = os.path.join(result_dir, best_folder, 'classifier_best.pth.tar')
    #     student_classifier = load_model(student_classifier, classifier_checkpoint)

    # switch to train mode
    teacher_model.eval()
    student_model.eval()
    tf_block.eval()
    # student_classifier.eval()

    # additional function
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    pdist = nn.PairwiseDistance()
    sampler = TupleSampler('npair')

    total_distance_mmd = []
    total_distance_mmd_rbf = []
    total_distance_pdist = []
    total_distance_wad = []
    total_distance = []
    total_cross_negative_distance = []
    total_self_negative_distance = []

    # get data
    # for i, conc_data in enumerate(train_loader):
    for i, conc_data in enumerate(test_loader):
        h_data, v_data = conc_data
        azi, ele, targets = h_data
        v_inputs, _ = v_data

        # prepare input and target
        azi = azi.to(device, dtype=torch.float)
        ele = ele.to(device, dtype=torch.float)
        # v_inputs = torch.permute(v_inputs, (0, 2, 1, 3, 4)).to(device)
        v_inputs = v_inputs.to(device)
        targets = targets.to(device, dtype=torch.long)

        pseudo_targets, g_t = teacher_model(v_inputs)
        outputs, g_s = student_model(azi, ele)

        t1, t2, t3, t4 = g_t
        s1, s2, s3, s4 = g_s

        t_fmap = t4
        s_fmap = tf_block(s4)

        # if mode != 'kd_baseline':
        #     s_fmap = tf_block(s4)
        # else:
        #     s_fmap = t4

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

    print("Pairwise Distance: {:0.4f}".format(np.mean(total_distance_pdist)))
    print("Cross Negative Pairwise Distance: {:0.4f}".format(np.mean(total_cross_negative_distance)))
    print("Self Negative Pairwise Distance: {:0.4f}".format(np.mean(total_self_negative_distance)))
