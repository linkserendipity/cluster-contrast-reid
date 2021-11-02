from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname) # * 路径和文件名合起来 变成图片的绝对路径

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        # ! fname 仅仅是图片的名称.jpg 加上路径变成 fpath 再转成 'RGB' 经过 transform 之后变成img 这是个图片的格式???

        return img, fname, pid, camid, index
