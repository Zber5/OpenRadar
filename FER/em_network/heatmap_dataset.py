import os
import os.path
import numpy as np
import torch
os.chdir("../../")
from FER.utils import MapRecord


class HeatmapDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,
                 annotationfile_path: str):
        super(HeatmapDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self._parse_list()

    def _parse_list(self):
        self.map_list = [MapRecord(x.strip().split(), self.root_path) for x in open(self.annotationfile_path)]

    def __getitem__(self, index):
        record = self.map_list[index]
        return self._get(record)

    def _get(self, record):
        azi = np.load(record.path.format("azi"))
        azi = azi[record.onset:record.peak + 1]
        azi = np.expand_dims(azi, axis=0)

        ele = np.load(record.path.format("ele"))
        ele = ele[record.onset:record.peak + 1]
        ele = np.expand_dims(ele, axis=0)

        return azi, ele, record.label

    def _normalize(self, data, is_azi=True):
        azi_para = [73.505790, 3.681510]
        ele_para = [86.071959, 5.921158]
        if is_azi:
            return (data-azi_para[0])/azi_para[1]
        else:
            return (data - ele_para[0]) / ele_para[1]

    def __len__(self):
        return len(self.map_list)
