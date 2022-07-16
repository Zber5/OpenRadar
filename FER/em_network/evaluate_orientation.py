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
    write_log(classification_report(all_target, all_pred, target_names=emotion_list), logdir)
    return all_pred, all_target, all_outputs, test_loss


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

    # for mm in ['M1_0', 'M1_1', 'M1_2', 'M2_0', 'M2_1', 'M2_2', 'Distance_100cm']:
    # for mm in ['M3_0', 'M3_1', 'M3_2']:
    # for mm in ['Standing_S1', 'Standing_S2', 'Standing_S3']:
    # for mm in ['Ground_S1', 'Ground_S2', 'Ground_S3']:
    # for mm in ['Sitting_S1', 'Sitting_S2', 'Sitting_S3']:
    for mm in ['Distance_300cm', 'Distance_300cm_v1']:
        i = 0
        # for bf in ["Ours_oldData_Motion_v5_20220606-025052", "Ours_oldData_Motion_v2_20220602-192438",
        #            "Ours_oldData_Motion_v3_20220603-083125"]:
        # for bf in ["Ours_oldData_Standing_20220531-034336", "Ours_oldData_Standing_v1_20220606-234034"]:
        # for bf in ["Ours_oldData_Ground_20220531-182950", "Ours_oldData_Ground_v1_20220606-231925"]:
        # for bf in ["Ours_oldData_Sitting_20220531-161425", "Ours_oldData_Motion_v2_20220602-192438"]:
        for bf in ["Ours_oldData_Distance_300cm_20220607-184411", "Ours_oldData_Distance_300cm_v1_20220607-185855"]:

            path = dir_path("Evaluate_ours_{}_v{}".format(mm, i), result_dir)
            # i += 1

            # path = dir_path("Evaluate_ours_M2_1_v2", result_dir)
            # path = dir_path("Evaluate_oldData_ours_W3_v3", result_dir)
            # path = dir_path("Evaluate_olddata_ours_W_1_2_3", result_dir)
            # path = dir_path("Evaluate_ours_S_0_1_2_v1", result_dir)

            # subjects
            # best_folder = "Supervision_SUM_image2D_TransformerLarge_L3_CrossNpairLossWeighted_20220312-132254_best"
            # best_folder = "Ours_oldData_S_0_1_2_20220527-070014"
            # best_folder = "Ours_oldData_S_3_4_5_20220527-174224"
            # best_folder = "Ours_oldData_S_6_7_8_9_20220528-212151"
            # best_folder = "Ours_newData_S-2_20220528-061653"
            # best_folder = "Ours_oldData_S_6_7_8_9_20220528-212151"
            # best_folder = "Ours_oldData_S_6_7_8_9_v1_20220603-213111"

            # wearing
            # best_folder = "Ours_oldData_W_1_2_3_20220530-164918"

            # distance
            # best_folder = "Ours_oldData_Distance_70cm_20220601-005908"
            # best_folder = "Ours_oldData_Distance_150cm_20220601-013858"
            # best_folder = "Ours_oldData_Distance_200cm_20220601-150228"

            # independent
            # best_folder = "Ours_oldData_S_0_8_20220605-024135"
            # best_folder = "Ours_oldData_S_0_9_20220605-025434"
            # best_folder = "Ours_oldData_S_8_9_20220605-034314"

            # motion
            # best_folder = "Ours_oldData_Motion_v5_20220606-025052"
            # best_folder = "Ours_oldData_Motion_v2_20220602-192438"
            # best_folder = "Ours_oldData_Motion_v3_20220603-083125"
            best_folder = bf

            # pose

            # motion
            # save training config
            save_to_json(config, path['config'])

            # load data
            # videos_root = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\'
            # v_train_ann = os.path.join(videos_root, 'frames_train_S0_1_2.txt')
            # v_test_ann = os.path.join(videos_root, 'frames_test_S0_1_2.txt')
            #
            # heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"
            # h_train_ann = os.path.join(heatmap_root, "heatmap_train_S0_1_2.txt")
            # h_test_ann = os.path.join(heatmap_root, "heatmap_test_S0_1_2.txt")

            # videos_root = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\'
            # v_train_ann = os.path.join(videos_root, 'frames_test_S9.txt')
            # v_test_ann = os.path.join(videos_root, 'frames_test_S9.txt')
            #
            # heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"
            # h_train_ann = os.path.join(heatmap_root, "heatmap_test_S9.txt")
            # h_test_ann = os.path.join(heatmap_root, "heatmap_test_S9.txt")

            # load data
            # videos_root = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\'
            # v_train_ann = os.path.join(videos_root, 'frames_train_S3_4_5.txt')
            # v_test_ann = os.path.join(videos_root, 'frames_test_S3_4_5.txt')
            #
            # heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"
            # h_train_ann = os.path.join(heatmap_root, "heatmap_train_S3_4_5.txt")
            # h_test_ann = os.path.join(heatmap_root, "heatmap_test_S3_4_5.txt")

            # load data
            # videos_root = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\'
            # v_train_ann = os.path.join(videos_root, 'frames_train_S3_4_5.txt')
            # v_test_ann = os.path.join(videos_root, 'frames_test_S3_4_5.txt')
            #
            # heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"
            # h_train_ann = os.path.join(heatmap_root, "heatmap_train_S3_4_5.txt")
            # h_test_ann = os.path.join(heatmap_root, "heatmap_test_S3_4_5.txt")

            # wearing
            # videos_root = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\'
            # v_train_ann = os.path.join(videos_root, 'frames_test_W3.txt')
            # v_test_ann = os.path.join(videos_root, 'frames_test_W3.txt')
            #
            # heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"
            # h_train_ann = os.path.join(heatmap_root, "heatmap_test_W3.txt")
            # h_test_ann = os.path.join(heatmap_root, "heatmap_test_W3.txt")

            # load data
            # videos_root = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\'
            # v_train_ann = os.path.join(videos_root, 'frames_train_S6_7_8_9.txt')
            # v_test_ann = os.path.join(videos_root, 'frames_test_S6_7_8_9.txt')
            #
            # heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"
            # h_train_ann = os.path.join(heatmap_root, "heatmap_train_S6_7_8_9.txt")
            # h_test_ann = os.path.join(heatmap_root, "heatmap_test_S6_7_8_9.txt")

            # videos_root = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\'
            # v_train_ann = os.path.join(videos_root, 'frames_test_S2.txt')
            # v_test_ann = os.path.join(videos_root, 'frames_test_S2.txt')
            #
            # heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap_new_v1"
            # h_train_ann = os.path.join(heatmap_root, "heatmap_test_S2.txt")
            # h_test_ann = os.path.join(heatmap_root, "heatmap_test_S2.txt")

            # wearing
            # videos_root = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\'
            # v_train_ann = os.path.join(videos_root, 'frames_test_S6_7_8_9_v1.txt')
            # v_test_ann = os.path.join(videos_root, 'frames_test_S6_7_8_9_v1.txt')
            #
            # heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"
            # h_train_ann = os.path.join(heatmap_root, "heatmap_test_S6_7_8_9_v1.txt")
            # h_test_ann = os.path.join(heatmap_root, "heatmap_test_S6_7_8_9_v1.txt")

            # distance
            # videos_root = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\'
            # v_train_ann = os.path.join(videos_root, 'frames_test_Distance_200cm.txt')
            # v_test_ann = os.path.join(videos_root, 'frames_test_Distance_200cm.txt')
            #
            # heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"
            # h_train_ann = os.path.join(heatmap_root, "heatmap_test_Distance_200cm.txt")
            # h_test_ann = os.path.join(heatmap_root, "heatmap_test_Distance_200cm.txt")

            # motion
            videos_root = 'C:\\Users\\Zber\\Desktop\\Subjects_Frames\\'
            v_train_ann = os.path.join(videos_root, 'frames_test_{}.txt'.format(mm))
            v_test_ann = os.path.join(videos_root, 'frames_test_{}.txt'.format(mm))

            heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"
            h_train_ann = os.path.join(heatmap_root, "heatmap_test_{}.txt".format(mm))
            h_test_ann = os.path.join(heatmap_root, "heatmap_test_{}.txt".format(mm))

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

            train_loader = DataLoader(dataset_train, num_workers=4, pin_memory=True, batch_size=config['batch_size'])
            test_loader = DataLoader(dataset_test, num_workers=4, pin_memory=True, batch_size=config['batch_size'])

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
                # s_checkpoint = os.path.join(result_dir, best_folder, 'model_9.pth.tar')
                student_model.load_state_dict(torch.load(s_checkpoint))
            student_model = student_model.to(device)

            # student classifier
            student_classifier = Classifier_Transformer(input_dim=512 * 7 * 7, num_classes=config['num_classes'])
            if config['continue_learning']:
                classifier_checkpoint = os.path.join(result_dir, best_folder, 'classifier_best.pth.tar')
                # classifier_checkpoint = os.path.join(result_dir, best_folder, 'classifier_9.pth.tar')
                student_classifier.load_state_dict(torch.load(classifier_checkpoint))
            student_classifier = student_classifier.to(device)

            # tf block
            tf_block = TFblock_v4()
            if config['continue_learning']:
                block_checkpoint = os.path.join(result_dir, best_folder, 'block_best.pth.tar')
                # block_checkpoint = os.path.join(result_dir, best_folder, 'block_9.pth.tar')
                tf_block.load_state_dict(torch.load(block_checkpoint))
            tf_block = tf_block.to(device)

            criterion_test = nn.CrossEntropyLoss()

            # function
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
