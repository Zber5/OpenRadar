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


def pair_loss(x, y):
    pdist = nn.PairwiseDistance()
    x = at(x)
    y = at(y)
    # diff = pdist(x, y).pow(2).mean()
    diff = pdist(x, y).mean()
    return diff


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

    # results dir
    result_dir = "FER/results"
    # path = dir_path("Supervision_image2D_KD", result_dir)

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

    train_loader = DataLoader(dataset_train, num_workers=4, pin_memory=True, batch_size=config['batch_size'])
    test_loader = DataLoader(heatmap_test, num_workers=4, pin_memory=True, batch_size=config['batch_size'])

    # folder path
    best_folder = "Supervision_image2D_Transformer_power_20220214-155040"
    # best_folder = "Supervision_image2D_Transformer_20220213-100217"
    # best_folder = "Supervision_image2D_Transformer_stage2_updated_20220216-152105"
    # best_folder = "Supervision_image2D_Transformer_allBP_20220215-215409"

    # load teacher model
    fmodel = resnet18()
    cmodel = Classifier(num_classes=7)
    teacher_model = ResNetFull_Teacher(fmodel, cmodel)
    # checkpoint = os.path.join(result_dir, "Pretrained_ResNet_video_v1_20220122-002807", 'best.pth.tar')
    checkpoint = os.path.join(result_dir, "Pretrained_ResNet_video_v1_1_20220214-135834", 'model_best.pth.tar')
    # teacher_model = load_model(teacher_model, checkpoint)
    teacher_model.load_state_dict(torch.load(checkpoint))
    teacher_model = teacher_model.to(device)

    # student model
    student_model = ImageSingle_v1(num_classes=config['num_classes'], block=ImageDualNet_Single_v1,
                                   subblock=ImageNet_Large_v1)
    # s_checkpoint = os.path.join(result_dir, best_folder, 'student_model_best.pth.tar')
    s_checkpoint = os.path.join(result_dir, best_folder, 'model_best.pth.tar')
    # s_checkpoint = os.path.join(result_dir, best_folder, 'last_0.pth.tar')
    # s_checkpoint = os.path.join(result_dir, "Supervision_image2D_AT_20220122-004538", 'best.pth.tar')
    # s_checkpoint = os.path.join(result_dir, "Supervision_image2D_KD_20220204-234440", 'best.pth.tar')
    # student_model = load_model(student_model, s_checkpoint)
    student_model.load_state_dict(torch.load(s_checkpoint))
    student_model = student_model.to(device)

    # tf block
    tf_block = TFblock_v2(dim_in=512, dim_inter=128)
    # tf_block = TFblock(dim_in=512, dim_inter=128)
    block_checkpoint = os.path.join(result_dir, best_folder, 'block_best.pth.tar')
    # block_checkpoint = os.path.join(result_dir, best_folder, 'block_best.pth.tar')
    # block_checkpoint = os.path.join(result_dir, best_folder, 'block_0.pth.tar')
    # tf_block = load_model(tf_block, block_checkpoint)
    tf_block.load_state_dict(torch.load(block_checkpoint))
    tf_block = tf_block.to(device)

    # student classifier
    # student_classifier = Classifier_Transformer(input_dim=512, num_classes=config['num_classes'])
    # classifier_checkpoint = os.path.join(result_dir, best_folder, 'classifier_best.pth.tar')
    # classifier_checkpoint = os.path.join(result_dir, best_folder, 'classifier_0.pth.tar')
    # student_classifier = load_model(student_classifier, classifier_checkpoint)
    # student_classifier.load_state_dict(torch.load(classifier_checkpoint))
    # student_classifier = student_classifier.to(device)

    # switch to train mode
    teacher_model.eval()
    student_model.eval()
    tf_block.eval()

    m = nn.AdaptiveAvgPool2d((1, 1))

    pdist = nn.PairwiseDistance()

    total_distance_mmd = []
    total_distance_mmd_rbf = []
    total_distance_pdist = []
    total_distance_wad = []
    total_distance = []

    # get data
    for i, conc_data in enumerate(train_loader):
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

        dists = pair_loss(s_fmap, t_fmap)

        # t_fmap = m(t_fmap)
        # t_fmap = torch.mean(t_fmap, dim=1)
        # t_fmap = torch.squeeze(t_fmap)
        # t_fmap = F.normalize(t_fmap)
        t_fmap = at(t_fmap)

        # s_fmap = tf_block(s4)
        # s_fmap = m(s_fmap)
        #
        # s_fmap = F.normalize(s_fmap)
        # s_fmap = m(s_fmap)
        # s_fmap = torch.mean(s_fmap, dim=(2, 3))
        # s_fmap = torch.squeeze(s_fmap)
        # s_fmap = F.normalize(s_fmap)
        s_fmap = at(s_fmap)

        # MMD
        mmd = MMD(t_fmap, s_fmap, kernel="multiscale")
        # mmd = [MMD(t, s, kernel="multiscale") for t, s in zip(t_fmap, s_fmap)]

        mmd_rbf = MMD(t_fmap, s_fmap, kernel="rbf")

        # pdsit
        t_fmap = t_fmap.cpu().detach()
        s_fmap = s_fmap.cpu().detach()
        # dists = pdist(t_fmap, s_fmap)
        # dists = torch.mean(dists)

        t_fmap = t_fmap.numpy()
        s_fmap = s_fmap.numpy()

        # wasserstein_distance
        was_dist = [wasserstein_distance(t, s) for t, s in zip(t_fmap, s_fmap)]
        was_dist = np.mean(was_dist)

        # Euclidean distances
        np_dst = np.linalg.norm(t_fmap - s_fmap, axis=1)
        np_dst = np.mean(np_dst)

        total_distance.append(np_dst)
        total_distance_pdist.append(float(dists))
        total_distance_mmd.append(float(mmd.detach().cpu().numpy()))
        total_distance_mmd_rbf.append(float(mmd_rbf.detach().cpu().numpy()))
        total_distance_wad.append(was_dist)

    print("Eulucidan Distance: {:0.4f}".format(np.mean(total_distance)))
    print("Pairwise: {:0.4f}".format(np.mean(total_distance_pdist)))
    print("Wasserstein Distance: {:0.4f}".format(np.mean(total_distance_wad)))
    print("MMD: {:0.4f}".format(np.mean(total_distance_mmd)))
    print("MMD-rbf: {:0.4f}".format(np.mean(total_distance_mmd_rbf)))
