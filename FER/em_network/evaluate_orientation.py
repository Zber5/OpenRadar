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


def evaluate():
    # turn models to eval mode
    student_model.eval()
    student_classifier.eval()
    tf_block.eval()

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
            azi = azi.to(device, dtype=torch.float)
            ele = ele.to(device, dtype=torch.float)
            v_inputs = v_inputs.to(device)
            target = target.to(device, dtype=torch.long)

            # forward to model
            _, g_s = student_model(azi, ele)

            s1, s2, s3, s4 = g_s

            s_fmap = s4
            s_fmap = tf_block(s_fmap)
            s_input = s_fmap.view((s_fmap.size(0), -1))
            output = student_classifier(s_input)

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
    write_log(classification_report(all_target, all_pred, digits=4, target_names=emotion_list), logdir)
    return all_pred, all_target, all_outputs, test_loss


def overwrite_temp_file(tf_name, og_name, key):
    record_list = [x for x in open(og_name)]
    with open(tf_name, 'w') as f:
        for rd in record_list:
            f.write(rd.format(key))


def overwrite_temp_file_heatmap(tf_name, og_name, key):
    record_list = [x for x in open(og_name)]
    with open(tf_name, 'w') as f:
        for rd in record_list:
            f.write(rd.format(key, '{}'))


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

    dd = []
    str_format = '{}cm_{}d'
    for dist in ['30', '70', '100', '150', '200', '250', '300']:
        d = []
        for iid in ['30', '60', '90']:
            d.append(str_format.format(dist, iid))
        dd.append(d)

    # ddf = [
    #     'Ours_oldData_30cmD_20220704-165229',
    #     'Ours_oldData_70cmD_20220705-134724',
    #     'Ours_oldData_100cmD_20220705-141949',
    #     'Ours_oldData_150cmD_20220705-180704',
    #     'Ours_oldData_200cmD_20220705-185637',
    #     'Ours_oldData_250cmD_20220705-203706',
    #     'Ours_oldData_300cmD_20220705-212437'
    # ]

    ddf = [
        'Supervision_SUM_image2D_Ours_20220627-183753',
        'Ours_oldData_Distance_70cm_20220601-005908',
        'Ours_oldData_Sitting_20220531-161425',
        'Ours_oldData_Distance_150cm_20220601-013858',
        'Ours_oldData_Distance_200cm_20220601-150228',
        'Ours_oldData_Distance_250cm_20220602-020548',
        'Ours_oldData_Distance_300cm_20220607-184411'
    ]

    for dists, distfile in zip(dd, ddf):
        best_folder = distfile
        # create teacher model
        fmodel = resnet18()
        cmodel = Classifier(num_classes=7)
        teacher_model = ResNetFull_Teacher(fmodel, cmodel)
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

        # student classifier
        student_classifier = Classifier_Transformer(input_dim=512 * 7 * 7, num_classes=config['num_classes'])
        if config['continue_learning']:
            classifier_checkpoint = os.path.join(result_dir, best_folder, 'classifier_best.pth.tar')
            student_classifier.load_state_dict(torch.load(classifier_checkpoint))
        student_classifier = student_classifier.to(device)

        # tf block
        tf_block = TFblock_v4()
        if config['continue_learning']:
            block_checkpoint = os.path.join(result_dir, best_folder, 'block_best.pth.tar')
            tf_block.load_state_dict(torch.load(block_checkpoint))
        tf_block = tf_block.to(device)

        for doname in dists:
            path = dir_path("Evaluate_ours_{}".format(doname), result_dir)
            save_to_json(config, path['config'])

            videos_root = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\'
            f_og = 'frames_test_dd.txt'
            f_temp = 'frames_test_temp.txt'

            heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"
            h_og = 'heatmap_test_dd.txt'
            h_temp = 'heatmap_test_temp.txt'

            overwrite_temp_file(os.path.join(videos_root, f_temp), os.path.join(videos_root, f_og), doname)
            overwrite_temp_file_heatmap(os.path.join(heatmap_root, h_temp), os.path.join(heatmap_root, h_og), doname)

            # load data

            v_test_ann = os.path.join(videos_root, f_temp)
            h_test_ann = os.path.join(heatmap_root, h_temp)

            preprocess = transforms.Compose([
                ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # video datasets

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
            heatmap_test = HeatmapDataset(heatmap_root, h_test_ann, cumulated=config['cumulated_frame'],
                                          frame_cumulated=False, num_frames=config['h_num_frames'])

            dataset_test = ConcatDataset(heatmap_test, video_test)

            test_loader = DataLoader(dataset_test, num_workers=4, pin_memory=True, batch_size=config['batch_size'])

            # function
            criterion_test = nn.CrossEntropyLoss()
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            pdist = nn.PairwiseDistance()

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
