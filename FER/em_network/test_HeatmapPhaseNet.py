import torch
import torch.nn as nn
import numpy as np
import random
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from utils import device, AverageMeter, dir_path, write_log, accuracy, save_checkpoint
from models.Conv2D import ImagePhaseNet
from dataset import HeatmapDataset, ConcatDataset, PhaseDataset
from torch.utils.data import DataLoader
import os
import pandas as pd
from FER.utils import ROOT_PATH

os.chdir(ROOT_PATH)

# set seed, make result reporducable
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def evaluate(model, resdir, testloader):
    # load weights
    cmdir = os.path.join(resdir, 'cm.pdf')
    logdir = os.path.join(resdir, 'cm_log.txt')
    model_path = os.path.join(resdir, 'best.pth.tar')

    # load weights
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # test
    all_pred = []
    all_target = []
    test_loss = 0
    with torch.no_grad():
        for (azi, ele, target) in testloader:
            azi = torch.permute(azi, (0, 2, 1, 3, 4))
            ele = torch.permute(ele, (0, 2, 1, 3, 4))
            # prepare input and target to device
            azi = azi.to(device, dtype=torch.float)
            ele = ele.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.long)
            output = model(azi, ele)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = pred.cpu().numpy().flatten()
            target = target.cpu().numpy().flatten()
            all_pred = np.concatenate((all_pred, pred), axis=0)
            all_target = np.concatenate((all_target, target), axis=0)
    # print
    write_log(classification_report(all_target, all_pred, target_names=emotion_list), logdir)

    cm = confusion_matrix(all_target, all_pred)

    ax = sns.heatmap(cm, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Classes')
    ax.set_ylabel('Actual Classes');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(emotion_list)
    ax.yaxis.set_ticklabels(emotion_list)
    plt.savefig(cmdir)


if __name__ == "__main__":
    config = dict(num_epochs=60,
                  lr=0.0006,
                  lr_step_size=20,
                  lr_decay_gamma=0.2,
                  num_classes=7,
                  batch_size=16,
                  h_num_frames=100,
                  )

    emotion_list = ['Neutral', 'Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    # results dir
    result_dir = "FER/results"

    # heatmap root dir
    heatmap_root = "C:/Users/Zber/Desktop/Subjects_Heatmap"
    phase_root = "C:/Users/Zber/Desktop/Subjects_Phase"

    # annotation dir
    annotation_train = os.path.join(heatmap_root, "heatmap_annotation_train.txt")
    annotation_test = os.path.join(heatmap_root, "heatmap_annotation_test.txt")

    # dataloader
    heatmap_train = HeatmapDataset(heatmap_root, annotation_train)
    heatmap_test = HeatmapDataset(heatmap_root, annotation_test)
    phase_train = PhaseDataset(phase_root, annotation_train)
    phase_test = PhaseDataset(phase_root, annotation_test)

    dataset_train = ConcatDataset(heatmap_train, phase_train)
    dataset_test = ConcatDataset(heatmap_test, phase_test)
    train_loader = DataLoader(dataset_train, batch_size=config['batch_size'], num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size'], num_workers=4, pin_memory=True)

    # log path
    path = "C:/Users/Zber/Documents/Dev_program/OpenRadar/FER/results/sensor_heatmap_3dcnn_fusion_heatmap&phase_20220101-232921"

    # create model
    model = ImagePhaseNet(num_classes=config['num_classes'])

    # initialize critierion
    criterion = nn.CrossEntropyLoss()

    evaluate(model, path, test_loader)
