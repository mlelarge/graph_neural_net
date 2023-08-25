# This file should be run in the env provided in 
# https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/docs/01_benchmark_installation.md
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.data import LoadData

DATASET_NAME = "CIFAR10"
dataset = LoadData(DATASET_NAME)

testset_dense = [dataset.collate_dense_gnn([d]) for d in dataset.test]
testset_dense = [(g.squeeze(0), l) for (g,l) in testset_dense]

torch.save(testset_dense, '/home/mlelarge/data/superpixels/CIFAR10/cifar_test.pt')

valset = [dataset.collate_dense_gnn([d]) for d in dataset.val]
valset = [(g.squeeze(0), l) for (g,l) in valset]
torch.save(valset, '/home/mlelarge/data/superpixels/CIFAR10/cifar_val.pt')

trainset = [dataset.collate_dense_gnn([d]) for d in dataset.train]
trainset = [(g.squeeze(0), l) for (g,l) in trainset]
torch.save(trainset, '/home/mlelarge/data/superpixels/CIFAR10/cifar_train.pt')