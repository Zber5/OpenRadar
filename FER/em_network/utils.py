import json
import logging
import os
import shutil
import torch
from torch.utils.data import DataLoader, Dataset
import time
from pathlib import Path
import cv2
import torchvision.transforms as transforms
import datetime
import numpy as np
import random
import torch.nn.functional as F

# set seed, make result reporducable
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

EMO_CLASSES = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']


def Cos_similarity(x, y, dim=1):
    assert (x.shape == y.shape)

    if len(x.shape) >= 2:
        return F.cosine_similarity(x, y, dim=dim)
    else:
        return F.cosine_similarity(x.view(1, -1), y.view(1, -1))


# def Cos_similarity(x, y, dim=1):
#     assert (x.shape == y.shape)
#
#     return F.cosine_similarity(x.view(1, -1), y.view(1, -1))


def normalise_data(data):
    nor_data = (data - np.mean(data)) / np.std(data)
    return nor_data


def sensor_imag_data_loader(x, y, batch_size=50):
    dataset = SensorImagDataSet(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def sensor_imag_label_data_loader(x, y, label, batch_size=50):
    dataset = SensorImagDataSet(x, y, label)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def senor_heatmap_label_data_loader(azi, ele, label, batch_size=50):
    dataset = SensorHeatmapDataSet(azi, ele, label)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def video_data_loader(video_root, batchsize_train, batchsize_eval, num_data_each_class=80, test_portion=0.2):
    all_index = [i for i in range(num_data_each_class)]
    train_index = random.sample(all_index, int(num_data_each_class * (1 - test_portion)))
    test_index = list(set(all_index) - set(train_index))

    train_transform = transforms.Compose(
        [transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    train_dataset = VideoDataset(video_root, train_index, train_transform)

    val_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    val_dataset = VideoDataset(video_root, test_index, val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize_train, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batchsize_eval, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader


def np_to_eye(y, num_class):
    y = y.astype(int)
    y = y.reshape((-1, 1))
    length = len(y)
    temp = np.zeros((length, num_class))
    for i in range(length):
        temp[i, y[i]] = 1
    return temp


def twin_sliding_window(imag_length, sensor_length, imag_window=6, sensor_window=20, imag_step=3, sensor_step=10):
    imag_start = 0
    sensor_start = 0

    while (imag_start + imag_window) <= imag_length and (sensor_start + sensor_window) <= sensor_length:
        yield (imag_start, imag_start + imag_window), (sensor_start, sensor_start + sensor_window)
        imag_start += imag_step
        sensor_start += sensor_step


class SensorDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y


class SensorImagDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y


def _read_pics_as_numpy(pics_dir, length=85, transform=None):
    video = []

    pics = os.listdir(pics_dir)
    pics.sort()

    # for pic in pics():

    for l in range(length):

        pic_path = os.path.join(pics_dir, pics[i])
        img = cv2.imread(pic_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if transform is not None:
            img = transform(img)
        video.append(img)

    video = np.concatenate(video, axis=0)

    return video


def print_summary_txt(txt_file, model, input_size):
    from torchsummary import summary
    from contextlib import redirect_stdout
    with open(txt_file, 'a') as f:
        with redirect_stdout(f):
            summary(model, input_size, device=device)

    print("Model summary has been written.")


class VideoDataset(Dataset):
    def __init__(self, root, data_index, transform=None):
        emotion_list = {'Joy': 0, 'Surprise': 1, 'Anger': 2, 'Sadness': 3, 'Fear': 4, 'Disgust': 5}

        self.videos = []
        self.labels = []

        for emotion in emotion_list.keys():
            for index in data_index:
                video_path = os.path.join(root, "{}_{}".format(emotion, index))
                label = emotion_list[emotion]
                video = _read_pics_as_numpy(video_path)
                self.videos.append(video)
                self.labels.append(label)
        self.videos = np.concatenate(self.videos, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        x = self.videos[index]
        y = self.labels[index]
        return x, y


class SensorImagLabelDataSet(Dataset):
    def __init__(self, x, y, label):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.label = label.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        label = self.label[index]
        return x, y, label


class SensorHeatmapDataSet(Dataset):
    def __init__(self, hm_azi, hm_ele, label):
        self.azi = hm_azi.astype(np.float32)
        self.ele = hm_ele.astype(np.float32)
        self.label = label.astype(np.float32)

    def __len__(self):
        return len(self.azi)

    def __getitem__(self, index):
        azi = self.azi[index]
        ele = self.ele[index]
        label = self.label[index]
        return azi, ele, label


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def dir_path(dir_name, result_dir):
    res_dir = os.path.join(result_dir, (dir_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    path_to_model = os.path.join(res_dir, 'best_model.pt')
    path_to_log = os.path.join(res_dir, 'log.txt')
    path_to_timer_dic = os.path.join(res_dir, 'timer.json')
    path_to_metrics = os.path.join(res_dir, 'metrics.csv')
    path_to_config = os.path.join(res_dir, 'config.json')
    path_to_confusion_matrix = os.path.join(res_dir, 'confusion_matrix.pdf')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    dic = {
        'dir': res_dir,
        'log': path_to_log,
        'metrics': path_to_metrics,
        'model': path_to_model,
        'timer': path_to_timer_dic,
        'config': path_to_config,
        'cm': path_to_confusion_matrix,
    }
    return dic


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, checkpoint, name="", epoch=0):
    full_name = '{}_{}.pth.tar'
    filepath = os.path.join(checkpoint, full_name.format(name, epoch))
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)
    if is_best:
        torch.save(state, os.path.join(checkpoint, full_name.format(name, "best")))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def write_log(str, path_to_log):
    if isinstance(path_to_log, dict):
        for key in path_to_log:
            log = open(path_to_log[key], 'a')
            log.write(str)
            log.close()

    else:
        log = open(path_to_log, 'a')
        log.write(str)
        log.close()


def time_now():
    ISOTIMEFORMAT = '%d-%h-%Y-%H-%M-%S'
    string = '{:}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


class Logger(object):
    def __init__(self, log_dir, title, args=False):
        """Create a summary writer logging to log_dir."""
        self.log_dir = Path("{:}".format(str(log_dir)))
        if not self.log_dir.exists(): os.makedirs(str(self.log_dir))
        self.title = title
        self.log_file = '{:}/{:}_date_{:}.txt'.format(self.log_dir, title, time_now())
        self.file_writer = open(self.log_file, 'a')

        if args:
            for key, value in vars(args).items():
                self.print('  [{:18s}] : {:}'.format(key, value))
        self.print('{:} --- args ---'.format(time_now()))

    def print(self, string, fprint=True):
        print(string)
        if fprint:
            self.file_writer.write('{:}\n'.format(string))
            self.file_writer.flush()

    def write(self, string):
        self.file_writer.write('{:}\n'.format(string))
        self.file_writer.flush()


def model_parameters(_structure, _parameterDir):
    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):

            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)

    # gpu
    device = torch.device("cuda")
    # device = torch.device("cpu")
    model = _structure.to(device)

    # GPU
    # model = torch.nn.DataParallel(_structure).cuda()

    return model
