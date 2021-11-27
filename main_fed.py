#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy, random
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

random.seed(0)
if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        # transforms：常用的图片变换，例如裁剪、旋转等，Compose：串联多个图片变换的操作，Normalize：用均值和标准差归一化张量图像
        # ToTensor：convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
        # 0.1307和0.3081是mnist数据集的均值和标准差，因为mnist数据值都是灰度图，所以图像的通道数只有一个，因此均值和标准差各一个。
        # 要是imagenet数据集的话，由于它的图像都是RGB图像，因此他们的均值和标准差各3个，分别对应其R,G,B值。例如([0.485, 0.456,
        # 0.406], [0.229, 0.224, 0.225])就是Imagenet dataset的标准化系数（RGB三个通道对应三组系数）。数据集给出的均值和标准差系数
        # ，每个数据集都不同的，都是数据集提供方给出的。
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # torch.Size([60000, 28, 28])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        # torch.Size([10000, 28, 28])
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape
    print(img_size)

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        print("len_in", str(len_in))
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print("net_glob: {0}, model: {1}".format(net_glob, args.model))
    res = net_glob.train()
    # print(res)

    # copy weights
    # init global model weight
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # args.num_users -> 100
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        # w_locals -> 以 list 的形式汇总本地客户端的训练结果权重
        # loss_locals -> 以 list 的形式汇总本地客户端的训练损失
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        # args.frac: the fraction of clients: default 0.1
        # 每轮通信参与联邦训练的客户端数量
        m = max(int(args.frac * args.num_users), 1)
        # clients sample 过程
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            # local model train process
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            # w: 本地模型基于本地私有数据训练得到的权重
            # loss: 相应训练损失
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        # 每一轮通信根据本地训练结果, 通过中央服务器进行模型聚合
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        # 在下一个 round 时本地客户端继承训练结果
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()

    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')


    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Training accuracy: {:.2f} Training loss: {:.2f}".format(acc_train, loss_train))
    print("Testing accuracy: {:.2f} Testing loss: {:.2f}".format(acc_test, loss_test))
    plt.title("Training accuracy: %.2f Testing accuracy: %.2f" % (acc_train, acc_test))
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
