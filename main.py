import argparse
import time
import torch
import json
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from torch_geometric.loader import DataLoader
from copy import deepcopy

import utils
import model
from configs import *

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='datasets', help='Datadir')
parser.add_argument('--data', type=str, default='MCF-7', help='Dataset used')
parser.add_argument('--baseline', type=str, default='GCN', help='Baseline used')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--aug', type=int, default=50, help='Whether aug and per aug epoch')
parser.add_argument('--khigh', type=int, default=3, help='High eigen nums')
parser.add_argument('--klow', type=int, default=3, help='Low eigen nums')
parser.add_argument('--thigh', type=int, default=3, help='Number of combination for high eigen')
parser.add_argument('--tlow', type=int, default=3, help='Number of combination for low eigen')
parser.add_argument('--eps', type=float, default=1e-3, help='Minimal edge weight value')
parser.add_argument('--normal', type=float, default=0.05, help='Normal threshold')
parser.add_argument('--abnormal', type=float, default=0.95, help='Abnormal threshold')
parser.add_argument('--auglr', type=float, default=5e-2, help='Aug learning rate')
parser.add_argument('--augnepoch', type=int, default=10, help='Number of aug training epochs')
parser.add_argument('--warmepoch', type=int, default=50, help='Warm epoch then aug')
parser.add_argument('--times', type=int, default=1, help='Generate times of trainsize')
parser.add_argument('--device', type=int, default=1, help='Wether GPU')
args = parser.parse_args()

datadir = args.datadir
data = args.data
baseline = args.baseline
seed = args.seed
aug = args.aug
khigh = args.khigh
klow = args.klow
thigh = args.thigh
tlow = args.tlow
eps = args.eps
normal = args.normal
abnormal = args.abnormal
auglr = args.auglr
augnepoch = args.augnepoch
warmepoch = args.warmepoch
times = args.times
device = args.device

utils.set_seed(seed)

print("Model info:")
print(json.dumps(args.__dict__, indent='\t'))
device = torch.device("cuda:0" if torch.cuda.is_available() and device else "cpu")

dataset, trainset, valset, testset, unlabelset = utils.load_data(datadir, data, khigh, klow, aug)
baseline_config = baseline_configs[baseline]
baseline_config['in_dim'] = dataset[0].x.shape[1]
baseline_config['out_dim'] = 2
baseline_config['baseline'] = baseline

normalcount, abnormalcount = utils.normal_abnormal_count(trainset)
info = {'normal': normalcount, 
        'abnormal': abnormalcount}
num_nodes = [graph.num_nodes for graph in dataset]
info['num_atom_supp'] = int(np.median(num_nodes))

framework = framework_dict[baseline](device, baseline_config, info)
auggad, aug_optimizer, aug_cos, aug_entropy = model.build_aug(thigh, tlow, eps, device, auglr, info)

batchsize = baseline_config['batchsize']
train_loader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(valset, batch_size=batchsize, shuffle=False)
unlabel_loader = DataLoader(unlabelset, batch_size=batchsize, shuffle=False)
test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False)

framework.add_origin_train_loader(train_loader)
if 'pretrain' in baseline_config:
    dataset_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    framework.pretrain(data, dataset_loader)

nepoch = baseline_config['nepoch']

for epoch in range(nepoch):
    if aug and (epoch + 1) > warmepoch and epoch % aug == 0:
        auggad = model.train_aug(framework, auggad, augnepoch, train_loader, val_loader, aug_optimizer, aug_cos, aug_entropy, device)
        aug_optimizer = optim.Adam(auggad.parameters(), lr=auglr)
        new_trainset = model.gen_data(framework, auggad, trainset, unlabelset, unlabel_loader, normal, abnormal, batchsize, times, device)
        train_loader = DataLoader(new_trainset, batch_size=batchsize, shuffle=True)

    train_start = time.time()
    train_loss = framework.train(train_loader)
    train_end = time.time()

    val_start = time.time()
    val_AUROC, val_AUPRC, val_MF1 = framework.val(val_loader, epoch)
    val_end = time.time()

test_start = time.time()
test_AUROC, test_AUPRC, test_MF1 = framework.test(test_loader)
test_end = time.time()

print("Epoch: {}, test AUROC: {:.4f}, AUPRC: {:.4f}, MF1: {:.4f}, time cost: {:.2f}".format(framework.best_epoch, test_AUROC, test_AUPRC, test_MF1, test_end - test_start))
