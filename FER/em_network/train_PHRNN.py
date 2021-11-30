import torch
import torch.nn as nn
import numpy as np
import random
import time
import os
from sklearn.model_selection import train_test_split

from utils import device, AverageMeter, dir_path, write_log, \
    sensor_imag_data_loader, accuracy
from models.PHRNN import PHRNN

os.chdir('//')

# set seed, make result reporducable
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# facial parts slices
# alignment landmark
face_parts = {
    'eyebrow': slice(17, 27),
    'eye': slice(36, 48),
    'nose': slice(27, 36),
    'lips': slice(48, 68),
}


def train(model, data_loader, criterion, optimizer, epoch=0, to_log=None, print_freq=5):
    # create Average Meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_loss = []

    # switch to train mode
    model.train()
    # record start time
    start = time.time()

    for i, (inputs, target) in enumerate(data_loader):
        # prepare input and target
        inputs = inputs.to(device)
        # target = target.type(torch.LongTensor)
        target = target.long()
        target = target.to(device)

        eyebrow, eye, nose, mouth = split_parts(inputs)
        eyebrow = reshape_parts(eyebrow)
        eye = reshape_parts(eye)
        nose = reshape_parts(nose)
        mouth = reshape_parts(mouth)

        # measure data loading time
        data_time.update(time.time() - start)

        # zero the parameter gradients
        optimizer.zero_grad()

        # gradient and do SGD step
        output = model(eyebrow, eye, nose, mouth)
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
        for (inputs, target) in test_loader:
            target = target.long()
            inputs, target = inputs.to(device), target.to(device)

            eyebrow, eye, nose, mouth = split_parts(inputs)
            eyebrow = reshape_parts(eyebrow)
            eye = reshape_parts(eye)
            nose = reshape_parts(nose)
            mouth = reshape_parts(mouth)

            output = model(eyebrow, eye, nose, mouth)
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


def scale_range(input, min=0, max=1):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input


def load_data(landmark_dir):
    x = np.zeros((num_data, num_frames, num_landmarks, n_dim))
    y = np.zeros((num_data))

    # load numpy data
    i = 0
    for label in range(1, 8):
        landmark_label_dir = os.path.join(landmark_dir, str(label))

        for npy_path in os.listdir(landmark_label_dir):
            d = np.load(os.path.join(landmark_label_dir, npy_path))
            x[i] = d
            y[i] = label - 1
            i += 1
    return x, y


def landmark_normalization(landmark):
    # nose point
    nose_index = 33

    for data_index in range(num_data):
        for frame_index in range(num_frames):
            frame_landmark = landmark[data_index, frame_index]
            nose_landmark = frame_landmark[nose_index]
            landmark[data_index, frame_index] = (frame_landmark-nose_landmark)/ np.std(frame_landmark, axis=0)

    return landmark


def reshape_parts(data):
    batch_size, frame_size = data.size()[0], data.size()[1]
    data_view = data.view((batch_size, frame_size, -1))
    return data_view


def split_parts(data):
    eyebrow = data[:, :, face_parts['eyebrow'], :]
    eye = data[:, :, face_parts['eye'], :]
    nose = data[:, :, face_parts['nose'], :]
    lips = data[:, :, face_parts['lips'], :]

    return eyebrow, eye, nose, lips


if __name__ == "__main__":

    N_EPOCHS = 100
    LR = 0.0005
    BATCH_SIZE = 32
    num_classes = 6
    num_data = 327
    num_frames = 30
    # num_frames = 20
    num_landmarks = 68
    n_dim = 2

    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    emotion_classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

    # results dir
    result_dir = "FER/results"
    # landmark_dir = "G:/My Drive/mmWave/mmWave-Emotion/mmWave Vision Datasets/CK+/CK+/emotion_images"
    landmark_dir = "G:/My Drive/mmWave/mmWave-Emotion/mmWave Vision Datasets/CK+/CK+/alignment_landmarks_L30/"
    # landmark_dir = "G:/My Drive/mmWave/mmWave-Emotion/mmWave Vision Datasets/CK+/CK+/alignment_landmarks_L20/"

    # model configure
    model_config = {
        'eye_size': (face_parts['eye'].stop - face_parts['eye'].start) * n_dim,
        'eyebrow_size': (face_parts['eyebrow'].stop - face_parts['eyebrow'].start) * n_dim,
        'nose_size': (face_parts['nose'].stop - face_parts['nose'].start) * n_dim,
        'mouth_size': (face_parts['lips'].stop - face_parts['lips'].start) * n_dim,
        'h1_size': 30,
        'h2_size': 30,
        'h3_size': 60,
        'h4_size': 60,
        'h5_size': 90,
        'h6_size': 90,
        'total_length': num_frames,
        'num_classes': len(emotion_classes)
    }

    # load data
    x, y = load_data(landmark_dir)
    # merge numpy file with same lable

    x = landmark_normalization(x)

    # Landmark normalization

    # azi_data_path = "demo/Emotion/data/Heatmap_D0_S1_L0_B4-14_I0-80_azi.npy"
    # ele_data_path = "demo/Emotion/data/Heatmap_D0_S1_L0_B4-14_I0-80_ele.npy"
    # label_path = 'demo/Emotion/data/sensor_b8r3_c5_y.npy'
    # label_path_1 = 'demo/Emotion/data/sensor_b8r3_c5_y_s40_e80.npy'

    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25, stratify=y)

    train_loader = sensor_imag_data_loader(x_train, y_train, batch_size=BATCH_SIZE)
    test_loader = sensor_imag_data_loader(x_test, y_test, batch_size=np.shape(x_test)[0])

    # log path
    path = dir_path("vision_landmark_PHRNN_alignment_landmark_normalized_nobias", result_dir)

    # create model
    model = PHRNN(**model_config)
    model = model.to(device)

    # initialize critierion and optimizer
    # could add weighted loss e.g. pos_weight = torch.ones([64])
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)

    metrics_dic = {
        'loss': [],
        'precision': [],
        'recall': []
    }

    for epoch in range(N_EPOCHS):
        train_loss = train(model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch,
                           to_log=path['log'])
        test_loss, acc = test(model, test_loader=test_loader, criterion=criterion, to_log=path['log'])

        lr_scheduler.step()

        metrics_dic['loss'].append(test_loss)
        metrics_dic['precision'].append(acc)
        # metrics_dic['recall'].append(rec)

    # save final model
    torch.save(model.state_dict(), path['model'])

    # save metrics dic
