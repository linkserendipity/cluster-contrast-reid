# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory # @ 记忆字典?
from clustercontrast.trainers import ClusterContrastTrainer # @ contrast cluster 训练?
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler # @ sampler?
from clustercontrast.utils.data.preprocessor import Preprocessor            # @ 预处理???
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance     # ! jaccard 有什么特别的???

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name) # ! /mnt/SSD/ls/data + / + market or dukemtmcreid or ...
    dataset = datasets.create(name, root)
    return dataset
    # dataset.append((img_path, pid, camid))

def get_train_loader(args, dataset, height, width, batch_size, workers, 
                     num_instances, iters, trainset=None): # * 比test_loader多了num_instances 和 iters
                                            # ! NOTE trainset=pseudo_labeled_dataset 用到的是伪标签的数据集啊哈哈哈
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5), #?
        T.Pad(10),
        T.RandomCrop((height, width)), #?
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]) # * 还要加上随机擦除!!!!!!
    ])
    #? 一定要sorted吗?
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0     # =16
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances) # REVIEW RandomMultipleGallerySampler 这个函数做啥用的 
        # ! GCN 那里用的是 RandomIdentitySampler 这里用了Multiple????
    else:
        sampler = None
    train_loader = IterLoader(   # REVIEW  IterLoader( 又是啥 用到了sampler 不shuffle 用到了drop_last???? length?
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
                    # NOTE 原来在这里用到了 length=iters 
    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # ! NOTE 这个和train loader的区别在哪???
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])
    # * testset没设置 则把test的query和gallery放进来
    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery)) # NOTE
    # TODO line157 testset=sorted(dataset.train) 就是把训练集升序排列一下

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
        # * Preprocessor: return img, fname, pid, camid, index

    return test_loader


def create_model(args): # FIXME num_features=args.features 这是啥??
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0, pooling_type=args.pooling_type) #TODO? num_classes=0?????
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model) #NOTE 先放到cuda里再并行????
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        # seed设置

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = round(time.monotonic())

    cudnn.benchmark = True
    ##### * 保存到子文件夹里 避免覆盖 （args.logs_dir 都改成 data_save_path） #!!
    if args.logs_dir==osp.join(working_dir, 'logs'):
        data_save_path = osp.join(args.logs_dir, args.dataset) # 指定好了就不会运行这行
    else:
        data_save_path = args.logs_dir

    sys.stdout = Logger(osp.join(data_save_path, 'log_{}.txt'.format(args.dataset))) #!!
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)  # * args.dataset表示dataset的名字
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    # 

    # Create model
    model = create_model(args) # ? num_classes == 0 ???
    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    # ! 这个params默认的吗?? Adam lr weight_decay要改吗
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1) # ! StepLR

    # Trainer
    trainer = ClusterContrastTrainer(model) #! NOTE resent50 训练函数

    for epoch in range(args.epochs):
        with torch.no_grad(): # NOTE 测试 不需要回传 参数???!!!! 
            # 第一步直接用 pretrained resnet50 提取特征 然后作 compute_jaccard_distance()? 第0个epoch才需要 DBScan 生成伪标签?
            #! STUB  GCN 取代了 DBSan 作聚类生成伪标签 然后用cross-entropy loss训练GCN
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))
                                             # TODO testset=sorted(dataset.train) 怎么用的 用的是全部的train吗???
            # return img, fname, pid, camid, index

            features, _ = extract_features(model, cluster_loader, print_freq=50)  # return features, labels
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0) #REVIEW 拼接 train 的features unsqueeze在外面加一个中括号 即增加一个维度
            #! TODO  直接用stack不行吗 torch.stack([features[f] for f, _, _ in sorted(dataset.train)], dim=0)
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2) # NOTE jaccard Distance?
            # k1=30 k2=6 faiss_rerank.py???????
            if epoch == 0: # * DBSCAN 只在第一代使用??
                # DBSCAN cluster 
                eps = args.eps # 0.6?????
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1) # * REVIEW DBSCAN

            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)  ##!!!!!!!!! REVIEW  产生pseudo_labels 全是train的
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0) ## * 很关键 伪标签的种类数

            # print("epoch: {} \n pseudo_labels: {}".format(epoch, pseudo_labels.tolist()[:100]))

        # !generate new dataset and calculate cluster centers
        # @ 装饰器????
        @torch.no_grad()
        def generate_cluster_features(labels, features): # TODO 生成cluster 特征?
            centers = collections.defaultdict(list) # * 该函数返回一个类似字典的对象
            for i, label in enumerate(labels):
                if label == -1: # -1的干扰图片去掉 这些用不到了
                    continue
                centers[labels[i]].append(features[i]) # * 伪标签形成的 center cluster 加入对应的特征向量

            centers = [ # NOTE torch.stack 把 每个cluster内的特征都拼接起来 然后 #!取平均值 就变成 c0 c1 ... 再连成list
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys()) 
            ]

            centers = torch.stack(centers, dim=0) # * 加一维 连成一个大的 tensor
            return centers

        cluster_features = generate_cluster_features(pseudo_labels, features) # * 由伪标签和特征 得到所有 cluster 的中心特征 
        del cluster_loader, features # 删掉cluster和features 下一个epoch再生成新的 FIXME 每一代生成的cluster_loader不都一样吗??????? features倒是因为model更新了所以会不一样

        # *** Create hybrid memory                      temperature for scaling contrastive loss // use_hard?
        memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp, # FIXME 公式(2)!!!!
                               momentum=args.momentum, use_hard=args.use_hard).cuda()   # * temp=0.05 momentum=0.2  放到cuda里
        memory.features = F.normalize(cluster_features, dim=1).cuda() # NOTE normalize啥东西??

        trainer.memory = memory #ANCHOR memory 记忆字典?!

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1: # -1 表示没有生成聚类伪标签的图片
                pseudo_labeled_dataset.append((fname, label.item(), cid)) # NOTE 贴好伪标签的训练集 图片地址和label,cameraID对应 

        print('==> Statistics for training epoch {}: {} clusters'.format(epoch, num_cluster))
        # * 开始用伪标签训练集 给 Resnet50 reid训练
        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,  # iters=400
                                        trainset=pseudo_labeled_dataset)

        train_loader.new_epoch() # NOTE 这是做啥用的?? IterLoader(  self.iter = iter(self.loader)
        # * 训练resnet50啦
        trainer.train(epoch, train_loader, optimizer, #! NOTE len(train_loader)= 多少???
                      print_freq=args.print_freq, train_iters=len(train_loader))
        # 每5个epoch测试一次 用query和gallery 
        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False) # * 只返回mAP 用于计算 is_best 保存最好的模型
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(data_save_path, 'checkpoint.pth.tar'))  #!!

            print('\n * Finished epoch {:3d}  model \033[0;32;1mmAP: {:5.1%}\033[0m  best: {:5.1%}{}\n'.   
                  format(epoch, mAP, best_mAP, ' *' if is_best else '')) #!!!!

        lr_scheduler.step() # NOTE 这里作 lr.step()???

    # 开始测试 best model 的指标
    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(data_save_path, 'model_best.pth.tar'))  #!!
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True) #*  cmc_flag=True 的情况 return cmc_scores['market1501'], mAP

    end_time = round(time.monotonic())
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid', #?
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,   ###!!! ?
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6, # TODO 改成0.4??
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0) # FIXME create model 怎么弄的???? 怎么默认是0???
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100) #!!! 改成100好了
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__)) ## !!!!!! working_dir是cluster_contrast_train_usl.py 所在的examples文件夹？？？
    # 把 data-dir 的working_dir 改成 /mnt/SSD/ls
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join('/mnt/SSD/ls', 'data')) #!!
                        # default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    main()
    # 先parser 再主函数

    # TODO 报错发邮件 跑完了结果发邮件

    
