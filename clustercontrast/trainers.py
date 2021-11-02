from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter


class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder # * resnet50
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400): # 默认 train_iters=400 = iters
        self.encoder.train() # 表明是训练过程 eval() 测试时用
        # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters): # @ NOTE GCN 用的是for i, inputs in enumerate(data_loader):
            # load data
            inputs = data_loader.next() #! NOTE next()?? 
            # ! self.iter = iter(self.loader)             return next(self.iter)

            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            loss = self.memory(f_out, labels) # ! loss = F.cross_entropy(outputs, targets) # 交叉熵 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i) % (print_freq * 2) == 0: # ! 去掉 + 1  50*2=100
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i , len(data_loader), #!
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))
    # * img pid index 放到cuda里面去
    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()
        # * img, fname, pid, camid, index
        # ! fname 加上路径变成 fpath 再转成 'RGB' 经过 transform 之后变成img

    def _forward(self, inputs):
        return self.encoder(inputs)
        #* 把图片提取出特征来

