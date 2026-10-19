#!/usr/bin/python
# -*- coding:utf8 -*-
from __future__ import print_function
import argparse
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import grad
import time
from torch import autograd
from PIL import ImageFile
import scipy as sp
import scipy.stats
import pdb

import sys
sys.dont_write_bytecode = True
sys.path.append((os.path.dirname(os.path.abspath(__file__))))

# ============================ Data & Networks =====================================
from dataset.datasets_csv import Imagefolder_csv
import models.network as DN4Net


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/Datasets/miniImageNet--ravi', help='/miniImageNet')
parser.add_argument('--data_name', default='miniImageNet', help='miniImageNet|StanfordDog|StanfordCar|CubBird')
parser.add_argument('--mode', default='train', help='train|val|test')
parser.add_argument('--outf', default='./results/DN4')
parser.add_argument('--resume', default='', type=str, help='path to the lastest checkpoint (default: none)')
parser.add_argument('--basemodel', default='Conv64F', help='Conv64F|ResNet256F')
parser.add_argument('--workers', type=int, default=8)
#  Few-shot parameters  #
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--episodeSize', type=int, default=1, help='the mini-batch size of training')
parser.add_argument('--testepisodeSize', type=int, default=1, help='one episode is taken as a mini-batch')
parser.add_argument('--epochs', type=int, default=30, help='the total number of training epoch')
parser.add_argument('--episode_train_num', type=int, default=10000, help='the total number of training episodes')
parser.add_argument('--episode_val_num', type=int, default=1000, help='the total number of evaluation episodes')
parser.add_argument('--episode_test_num', type=int, default=1000, help='the total number of testing episodes')
parser.add_argument('--way_num', type=int, default=5, help='the number of way/class')
parser.add_argument('--shot_num', type=int, default=1, help='the number of shot')
parser.add_argument('--query_num', type=int, default=15, help='the number of queries')
parser.add_argument('--neighbor_k', type=int, default=3, help='the number of k-nearest neighbors')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate, default=0.005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='the number of gpus')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 100)')
opt = parser.parse_args()
opt.cuda = True
cudnn.benchmark = True


model = DN4Net.define_DN4Net(which_model=opt.basemodel, num_classes=opt.way_num, neighbor_k=opt.neighbor_k, norm='batch',
	init_type='normal', use_gpu=opt.cuda)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))
model_path = '/workspace/papers/2/DN4/results/DN4_MSTAR_Conv64F_5Way_1Shot_K3/model_best.pth.tar'
checkpoint = torch.load(model_path)
#print(checkpoint)
print(type(checkpoint))
print(checkpoint.keys())
print(checkpoint['epoch_index'])
print(checkpoint['best_prec1'])
print(type(checkpoint['state_dict']))
print(checkpoint['state_dict'].keys())
epoch_index = checkpoint['epoch_index']
best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
print(111)
