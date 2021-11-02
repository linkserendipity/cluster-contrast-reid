from __future__ import absolute_import

from .base_dataset import BaseDataset, BaseImageDataset
from .preprocessor import Preprocessor


class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if self.length is not None:
            return self.length #  指定好了的 length=iters =400!!!!! 
            #! NOTE 训练resnet50的时候 trainer.train(  train_iters=len(train_loader)=length=iters=400!!!

        return len(self.loader)
    # ! 这个new_epoch是做啥用的??
    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)
