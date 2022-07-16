import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import parameters as iap
import random
from utils import normalise_data

ia.seed(123)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class MapRecord(object):
    """
    Helper class for class VideoFrameDataset. This class
    represents a video sample's metadata.

    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 1) The first
             element is the path to the video sample's frames excluding
             the root_datapath prefix 2) The  second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
             4) The fourth element is the label index.
             5) any following elements are labels in the case of multi-label classification
    """

    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    @property
    def relative_path(self):
        return self._data[0]

    @property
    def num_frames(self):
        if self.offset == -1:
            return self.peak - self.onset + 1  # +1 because end frame is inclusive
        else:
            return self.offset - self.onset + 1  # +1 because end frame is inclusive

    @property
    def onset(self):
        return int(self._data[2])

    @onset.setter
    def onset(self, value):
        self._data[2] = value

    @property
    def peak(self):
        return int(self._data[3])

    @peak.setter
    def peak(self, value):
        self._data[3] = value

    @property
    def offset(self):
        return int(self._data[4])

    @offset.setter
    def offset(self, value):
        self._data[4] = value

    @property
    def width_err(self):
        return int(self._data[5])

    @property
    def height_err(self):
        return int(self._data[6])

    @property
    def index_err(self):
        return int(self._data[7])

    @property
    def label(self):
        return int(self._data[1])


class HeatmapDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 cumulated=False,
                 frame_cumulated=False,
                 aug=False,
                 num_frames=100):
        super(HeatmapDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self._parse_list()
        # self.crop_azi = np.s_[:, 20:70, 3:8]
        # self.crop_ele = np.s_[:, 20:70, 3:8]

        # large
        # self.crop_azi = np.s_[:, 45:136, :]
        # self.crop_ele = np.s_[:, 45:136, :]

        # small
        # self.crop_azi = np.s_[:, 30:60, 3:8]
        # self.crop_ele = np.s_[:, 30:60, 3:8]

        # og
        self.crop_azi = np.s_[:, :, :]
        self.crop_ele = np.s_[:, :, :]

        self.diff = False
        self.cumulated = cumulated
        self.frame_cumulated = frame_cumulated
        self.num_frames = num_frames
        self.total = 100
        self.num_cumulated = self.total // num_frames
        self.sum_axis = 0 if self.num_frames == self.total else 1
        self.aug = aug
        self.pad = False
        self.azi_para = None
        self.ele_para = None

        blurer_gaussian = iaa.GaussianBlur(0.5)  # blur images with a sigma between 0 and 3.0
        blurer_mean = iaa.AverageBlur(k=3)  # blur image using local means with kernel sizes between 2 and 7
        blurer_median = iaa.MedianBlur(k=3)  # blur image using local medians with kernel sizes between 2 and 7
        blurer_motion = iaa.MotionBlur(k=5, angle=(5, 20))
        gaussian_noise = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
        avg_pooling = iaa.AveragePooling(2)
        # crop = iaa.Crop(percent=(0.1, 0.1))
        crop = iaa.CropAndPad(percent=(0.1, 0.1), pad_mode=ia.ALL)
        # self.aug_option = [blurer_gaussian, blurer_motion, gaussian_noise, avg_pooling, crop, None]
        # self.aug_option = [blurer_gaussian, blurer_mean, crop]
        self.aug_option = [blurer_gaussian, crop, blurer_mean, avg_pooling]

    def _parse_list(self):
        # self.map_list = [MapRecord(x.strip().split(), self.root_path) for x in open(self.annotationfile_path)]
        self.map_list = [MapRecord(x.strip().split(), self.root_path) for x in open(self.annotationfile_path)]

    def __getitem__(self, index):
        record = self.map_list[index]
        return self._get(record)

    def _get(self, record):
        azi = np.load(record.path.format("azi"))

        # replace -inf with 0
        # azi[np.isneginf(azi)] = 0

        azi = azi[self.crop_azi]
        if self.diff:
            record.onset = record.onset - 1
        azi = azi[record.onset:record.peak + 1]
        azi = self._normalize(azi, is_azi=True)
        if self.diff:
            azi = np.diff(azi, axis=0)
        if self.cumulated:
            azi = np.split(azi, self.num_cumulated, axis=0)
            azi = np.squeeze(azi)
            azi = np.sum(azi, axis=self.sum_axis)
            azi /= self.num_cumulated
        if self.frame_cumulated:
            azi = np.cumsum(azi, axis=0)
            azi_seed = np.expand_dims(np.arange(1, 101, 1), axis=(1, 2))
            azi = azi / azi_seed
        azi = np.expand_dims(azi, axis=0)

        ele = np.load(record.path.format("ele"))

        # replace -inf with 0
        # ele[np.isneginf(ele)] = 0

        ele = ele[self.crop_ele]
        ele = ele[record.onset:record.peak + 1]
        ele = self._normalize(ele, is_azi=False)
        if self.diff:
            ele = np.diff(ele, axis=0)
        if self.cumulated:
            ele = np.split(ele, self.num_cumulated, axis=0)
            ele = np.squeeze(ele)
            ele = np.sum(ele, axis=self.sum_axis)
            ele /= self.num_cumulated
        if self.frame_cumulated:
            ele = np.cumsum(ele, axis=0)
            ele_seed = np.expand_dims(np.arange(1, 101, 1), axis=(1, 2))
            ele = ele / ele_seed
        ele = np.expand_dims(ele, axis=0)

        if self.aug:
            azi, ele = self._aug(azi, ele)

        if self.pad:
            azi = np.pad(azi, ((0, 0), (31, 30), (3, 2)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            ele = np.pad(ele, ((0, 0), (31, 30), (3, 2)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))

        return azi, ele, record.label

    def _normalize(self, data, is_azi=True):
        azi_para = [73.505790, 3.681510]
        ele_para = [86.071959, 5.921158]
        if is_azi:
            return (data - azi_para[0]) / azi_para[1]
        else:
            return (data - ele_para[0]) / ele_para[1]

    # def _normalize(self, data, is_azi=True):
    #     # if self.azi_para is None and is_azi:
    #     #     azi_mean = np.mean(data)
    #     #     azi_std = np.std(data)
    #     #     self.azi_para =
    #     # b = np.linalg.norm(data)
    #     # norm_data = data/b
    #     mean = np.mean(data)
    #     std = np.std(data)
    #     norm_data = (data-mean)/std
    #     return norm_data

    def __len__(self):
        return len(self.map_list)

    def _aug(self, azi, ele):
        if random.random() <= 0.3:
            return azi, ele
        aug_opt = random.sample(self.aug_option, 1)
        seq = iaa.Sequential(aug_opt)

        azi = seq(images=azi)
        ele = seq(images=ele)
        return azi, ele


class LandmarkDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 num_dim=2):
        super(LandmarkDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self._parse_list()
        self.num_dim = num_dim

    def _parse_list(self):
        self.map_list = [VideoRecord(x.strip().split(), self.root_path) for x in open(self.annotationfile_path)]

    def __getitem__(self, index):
        record = self.map_list[index]
        return self._get(record)

    def _get(self, record):
        landmark = np.load(record.path)
        landmark = landmark[record.start_frame:record.end_frame + 1]
        landmark = self._normalize(landmark)
        landmark = landmark[-1, :, :self.num_dim]
        return landmark, record.label

    def _normalize(self, data):
        nose_index = 2

        # position normalize
        for fid, frame in enumerate(data):
            nose_landmark = frame[nose_index]
            nose_landmark[2] = 0
            data[fid] = data[fid] - nose_landmark

        # scale normalize
        data[:, :, 0] = normalise_data(data[:, :, 0])
        data[:, :, 1] = normalise_data(data[:, :, 1])
        data[:, :, 2] = normalise_data(data[:, :, 2])

        return data

    def __len__(self):
        return len(self.map_list)


def video_from_array(array):
    images = []
    if array.ndim == 2:
        array = np.expand_dims(array, axis=0)

    for arr in array:
        # scale to 0 - 255
        new_arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
        new_arr = np.squeeze(new_arr)
        img = Image.fromarray(new_arr)
        images.append(img)
    return images


class HeatmapDataset_Image(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 transform=None,
                 cumulated=False,
                 num_frames=100):
        super(HeatmapDataset_Image, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self._parse_list()
        # self.crop_azi = np.s_[:, 20:70, 3:8]
        # self.crop_azi = np.s_[:, 20:70, 3:8]
        self.crop_azi = np.s_[:, :, :]
        self.crop_ele = np.s_[:, :, :]
        self.diff = False
        self.cumulated = cumulated
        self.num_frames = num_frames
        self.total = 100
        self.num_cumulated = self.total // num_frames
        self.sum_axis = 0 if self.num_frames == self.total else 1
        self.transform = transform

    def _parse_list(self):
        self.map_list = [MapRecord(x.strip().split(), self.root_path) for x in open(self.annotationfile_path)]

    def __getitem__(self, index):
        record = self.map_list[index]
        return self._get(record)

    def _get(self, record):
        # to grey scale data 0-255

        azi = np.load(record.path.format("azi"))
        azi = azi[self.crop_azi]
        if self.diff:
            record.onset = record.onset - 1
        azi = azi[record.onset:record.peak + 1]
        azi = self._normalize(azi, is_azi=True)
        if self.diff:
            azi = np.diff(azi, axis=0)
        if self.cumulated:
            azi = np.split(azi, self.num_cumulated, axis=0)
            azi = np.squeeze(azi)
            azi = np.sum(azi, axis=self.sum_axis)
            azi /= self.num_cumulated
        azi = video_from_array(azi)

        ele = np.load(record.path.format("ele"))
        ele = ele[self.crop_ele]
        ele = ele[record.onset:record.peak + 1]
        ele = self._normalize(ele, is_azi=False)
        if self.diff:
            ele = np.diff(ele, axis=0)
        if self.cumulated:
            ele = np.split(ele, self.num_cumulated, axis=0)
            ele = np.squeeze(ele)
            ele = np.sum(ele, axis=self.sum_axis)
            ele /= self.num_cumulated
        ele = video_from_array(ele)

        if self.transform is not None:
            azi = self.transform(azi)
            ele = self.transform(ele)

        return azi, ele, record.label

    def _normalize(self, data, is_azi=True):
        azi_para = [73.505790, 3.681510]
        ele_para = [86.071959, 5.921158]
        if is_azi:
            return (data - azi_para[0]) / azi_para[1]
        else:
            return (data - ele_para[0]) / ele_para[1]

    def __len__(self):
        return len(self.map_list)


class PhaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 flatten=False):
        super(PhaseDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self._parse_list()
        self.diff = True
        self.flatten = flatten

    def _parse_list(self):
        self.map_list = [MapRecord(x.strip().split(), self.root_path) for x in open(self.annotationfile_path)]

    def __getitem__(self, index):
        record = self.map_list[index]
        return self._get(record)

    def _get(self, record):
        phase = np.load(record.path.replace("_{}", ""))
        if self.diff:
            phase = np.diff(phase, axis=-1)
        phase = phase[..., record.onset:record.peak + 1]
        if self.flatten:
            # phase = np.transpose(phase, (1, 2, 0))
            # phase = np.reshape(phase, (-1, phase.shape[2]))

            phase = np.transpose(phase, (2, 1, 0))
            phase = np.reshape(phase, (phase.shape[0], -1))

        return phase, record.label

    def __len__(self):
        return len(self.map_list)


class VideoRecord(object):
    """
    Helper class for class VideoFrameDataset. This class
    represents a video sample's metadata.

    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 1) The first
             element is the path to the video sample's frames excluding
             the root_datapath prefix 2) The  second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
             4) The fourth element is the label index.
             5) any following elements are labels in the case of multi-label classification
    """

    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])

    @property
    def path(self):
        return self._path

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive

    @property
    def start_frame(self):
        return int(self._data[1])

    @property
    def end_frame(self):
        return int(self._data[2])

    @property
    def label(self):
        # just one label_id
        if len(self._data) == 4:
            return int(self._data[3])
        # sample associated with multiple labels
        else:
            return [int(label_id) for label_id in self._data[3:]]


class VideoFrameDataset(torch.utils.data.Dataset):
    r"""
    A highly efficient and adaptable dataset class for videos.
    Instead of loading every frame of a video,
    loads x RGB frames of a video (sparse temporal sampling) and evenly
    chooses those frames from start to end of the video, returning
    a list of x PIL images or ``FRAMES x CHANNELS x HEIGHT x WIDTH``
    tensors where FRAMES=x if the ``ImglistToTensor()``
    transform is used.

    More specifically, the frame range [START_FRAME, END_FRAME] is divided into NUM_SEGMENTS
    segments and FRAMES_PER_SEGMENT consecutive frames are taken from each segment.

    Note:
        A demonstration of using this class can be seen
        in ``demo.py``
        https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch

    Note:
        This dataset broadly corresponds to the frame sampling technique
        introduced in ``Temporal Segment Networks`` at ECCV2016
        https://arxiv.org/abs/1608.00859.


    Note:
        This class relies on receiving video data in a structure where
        inside a ``ROOT_DATA`` folder, each video lies in its own folder,
        where each video folder contains the frames of the video as
        individual files with a naming convention such as
        img_001.jpg ... img_059.jpg.
        For enumeration and annotations, this class expects to receive
        the path to a .txt file where each video sample has a row with four
        (or more in the case of multi-label, see README on Github)
        space separated values:
        ``VIDEO_FOLDER_PATH     START_FRAME      END_FRAME      LABEL_INDEX``.
        ``VIDEO_FOLDER_PATH`` is expected to be the path of a video folder
        excluding the ``ROOT_DATA`` prefix. For example, ``ROOT_DATA`` might
        be ``home\data\datasetxyz\videos\``, inside of which a ``VIDEO_FOLDER_PATH``
        might be ``jumping\0052\`` or ``sample1\`` or ``00053\``.

    Args:
        root_path: The root path in which video folders lie.
                   this is ROOT_DATA from the description above.
        annotationfile_path: The .txt annotation file containing
                             one row per video sample as described above.
        num_segments: The number of segments the video should
                      be divided into to sample frames from.
        frames_per_segment: The number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders as described above.
        transform: Transform pipeline that receives a list of PIL images/frames.
        random_shift: Whether the frames from each segment should be taken
                      consecutively starting from the center of the segment, or
                      consecutively starting from a random location inside the
                      segment range.
        test_mode: Whether this is a test dataset. If so, chooses
                   frames from segments with random_shift=False.

    """

    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 num_segments: int = 3,
                 frames_per_segment: int = 1,
                 imagefile_template: str = 'img_{:05d}.jpg',
                 transform=None,
                 random_shift: bool = True,
                 test_mode: bool = False):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self._parse_list()
        self._sanity_check_samples()

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert('RGB')]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(), self.root_path) for x in open(self.annotationfile_path)]

    def _sanity_check_samples(self):
        for record in self.video_list:
            if record.num_frames <= 0 or record.start_frame == record.end_frame:
                print(f"\nDataset Warning: data sample {record.path} seems to have zero RGB frames on disk!\n")

    def _sample_indices(self, record):
        """
        For each segment, chooses an index from where frames
        are to be loaded from.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """

        segment_duration = (record.num_frames - self.frames_per_segment + 1) // self.num_segments
        if segment_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), segment_duration) + np.random.randint(
                segment_duration, size=self.num_segments)

        # edge cases for when a video has approximately less than (num_frames*frames_per_segment) frames.
        # random sampling in that case, which will lead to repeated frames.
        else:
            offsets = np.sort(np.random.randint(record.num_frames, size=self.num_segments))

        return offsets

    def _get_val_indices(self, record):
        """
        For each segment, finds the center frame index.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
             List of indices of segment center frames.
        """
        if record.num_frames > self.num_segments + self.frames_per_segment - 1:
            offsets = self._get_test_indices(record)

        # edge case for when a video does not have enough frames
        else:
            offsets = np.sort(np.random.randint(record.num_frames, size=self.num_segments))

        return offsets

    def _get_test_indices(self, record):
        """
        For each segment, finds the center frame index.

        Args:
            record: VideoRecord denoting a video sample
        Returns:
            List of indices of segment center frames.
        """

        tick = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets

    def __getitem__(self, index):
        """
        For video with id index, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations.

        Args:
            index: Video sample index.
        Returns:
            a list of PIL images or the result
            of applying self.transform on this list if
            self.transform is not None.
        """
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self._get(record, segment_indices)

    def _get(self, record, indices):
        """
        Loads the frames of a video at the corresponding
        indices.

        Args:
            record: VideoRecord denoting a video sample.
            indices: Indices at which to load video frames from.
        Returns:
            1) A list of PIL images or the result
            of applying self.transform on this list if
            self.transform is not None.
            2) An integer denoting the video label.
        """

        indices = indices + record.start_frame
        images = list()
        image_indices = list()
        for seg_ind in indices:
            frame_index = int(seg_ind)
            for i in range(self.frames_per_segment):
                seg_img = self._load_image(record.path, frame_index)
                images.extend(seg_img)
                image_indices.append(frame_index)
                if frame_index < record.end_frame:
                    frame_index += 1

        # sort images by index in case of edge cases where segments overlap each other because the overall
        # video is too short for num_segments*frames_per_segment indices.
        # _, images = (list(sorted_list) for sorted_list in zip(*sorted(zip(image_indices, images))))

        if self.transform is not None:
            images = self.transform(images)

        return images, record.label

    def __len__(self):
        return len(self.video_list)


class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``VideoFrameDataset``.
    """

    def forward(self, img_list):
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.

        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])
        # return torch.squeeze(torch.stack([transforms.functional.to_tensor(pic) for pic in img_list]))


def denormalize(video_tensor):
    """
    Undoes mean/standard deviation normalization, zero to one scaling,
    and channel rearrangement for a batch of images.
    args:
        video_tensor: a (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    """
    inverse_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inverse_normalize(video_tensor) * 255.).type(torch.uint8).permute(0, 2, 3, 1).numpy()
