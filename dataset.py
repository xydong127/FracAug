import argparse
import json
import time
from torch_geometric.transforms import AddSelfLoops
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import get_laplacian, to_dense_adj
import torch
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
import numpy as np
import random
import os

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=11, help='seed')
parser.add_argument('--datadir', type=str, default='datasets', help='datadir')
parser.add_argument('--data', type=str, default='MCF-7', help='data')
parser.add_argument('--khigh', type=int, default=0, help='Generate k largest eigens')
parser.add_argument('--klow', type=int, default=0, help='Generate k smallest eigens')
parser.add_argument('--reduce', type=int, default=0, help='Reduce x dimension')
parser.add_argument('--trainsz', type=float, default=0.01, help='train size')
parser.add_argument('--testsz', type=float, default=0.98, help='test size')
args = parser.parse_args()

seed = args.seed
datadir = args.datadir
data = args.data
khigh = args.khigh
klow = args.klow
reduce = args.reduce
trainsz = args.trainsz
testsz = args.testsz

assert trainsz + testsz < 1, "invalid train, val, and test size"

utils.set_seed(seed)
print("Generator info:")
print(json.dumps(args.__dict__, indent='\t'))

def reduce_x(datadir, name, dataset, reduce):
    savedir = os.path.join(datadir, name)
    savedir = os.path.join(savedir, "reduced" + str(reduce))

    os.makedirs(savedir, exist_ok=True)
    print("x dimension before reduction: {}".format(dataset.x.shape))
    x = sp.csr_matrix(dataset.x.numpy())
    x = x * x.T / x.shape[0]
    eigenvalues, eigenvectors = eigsh(x, k=reduce, maxiter=5000)
    eigenvalues = np.diag(eigenvalues)
    x = torch.from_numpy(np.matmul(eigenvectors, eigenvalues)).to(torch.float32)
    print("x dimension after reduction: {}".format(x.shape))
    torch.save(x, os.path.join(savedir, 'reduced_x.pt'))

def generate_eigens(datadir, name, dataset, k, which):
    savedir = os.path.join(datadir, name)
    savedir = os.path.join(savedir, which + str(khigh))
    
    os.makedirs(savedir, exist_ok=True)

    Us = []
    Es = []
    rowpadnum = 0
    for data in dataset:
        lap = get_laplacian(data.edge_index, normalization='sym', num_nodes=len(data.x))
        lap = to_dense_adj(edge_index=lap[0], edge_attr=lap[1])[0]
        adj = (torch.eye(len(lap)) - 0.5 * lap).numpy()
        eigenvalues, eigenvectors = eigsh(adj, k=k, which=which, maxiter=5000)
        rowpadnum = max(rowpadnum, len(eigenvectors))
        Us.append(eigenvectors)
        Es.append(eigenvalues)
    
    for i in range(len(Us)):
        colpadnum = 0
        if Us[i].shape[1] < k:
            colpadnum = k - Us[i].shape[1]
        Us[i] = np.pad(Us[i], [(0, rowpadnum - len(Us[i])), (0, colpadnum)], mode='constant')
        Us[i] = Us[i].reshape((1, ) + Us[i].shape)
        Es[i] = np.diag(np.pad(Es[i], (0, colpadnum), mode='constant'))
        Es[i] = Es[i].reshape((1, ) + Es[i].shape)

    Us = np.concatenate(Us, 0)
    Es = np.concatenate(Es, 0)
    Us = torch.from_numpy(Us)
    Es = torch.from_numpy(Es)
    Es = torch.where(Es < 0, 0, Es)

    print("Eigenvalues shape: {}, eigenvectors shape: {}".format(Es.shape, Us.shape))

    torch.save(Us, os.path.join(savedir, 'Us.pt'))
    torch.save(Es, os.path.join(savedir, 'Es.pt'))

dataset = TUDataset(root=datadir, name=data, transform=AddSelfLoops())

print("Average nodes: {:.2f}, average edges: {:.2f}, feature dim: {}".format(dataset.x.shape[0] / len(dataset), dataset.edge_index.shape[1] / 2 / len(dataset), dataset.x.shape[1]))

if reduce: 
    print("Start to generate reduced x for {}".format(data))
    s = time.time()
    reduce_x(datadir, data, dataset, reduce)
    e = time.time()
    print("Generating successfully, time cost: {:.2f}".format(e - s))

if khigh:
    print("Start to generate {} largest eigen value......".format(khigh))
    s = time.time()
    generate_eigens(datadir, data, dataset, khigh, 'LM')
    e = time.time()
    print("Generating successfully, time cost: {:.2f}".format(e - s))

if klow:
    print("Start to generate {} smallest eigen value......".format(klow))
    s = time.time()
    generate_eigens(datadir, data, dataset, klow, 'SM')
    e = time.time()
    print("Generating successfully, time cost: {:.2f}".format(e - s))

print("Start to generate train/val/test index......")
s = time.time()
labels = dataset.data.y.tolist()

NORMAL = 0
ABNORMAL = 1

normalinds = []
abnormalinds = []
errors = []

utils.set_seed(seed)

for i, label in enumerate(labels):
    if label == NORMAL:
        normalinds.append(i)
    elif label == ABNORMAL:
        abnormalinds.append(i)
    else:
        errors.append(i)

print("Normal nodes: {}, abnormal nodes: {}, abnormal rate: {:.4f}".format(len(normalinds), len(abnormalinds), len(abnormalinds) / len(labels)))

assert len(errors) == 0, "invalid labels"

train_normal = np.array(normalinds[: int(trainsz * len(normalinds))])
val_normal = np.array(normalinds[int(trainsz * len(normalinds)): int((1 - testsz) * len(normalinds))])
test_normal = np.array(normalinds[int((1 - testsz) * len(normalinds)): ])

train_abnormal = np.array(abnormalinds[: int(trainsz * len(abnormalinds))])
val_abnormal = np.array(abnormalinds[int(trainsz * len(abnormalinds)): int((1 - testsz) * len(abnormalinds))])
test_abnormal = np.array(abnormalinds[int((1 - testsz) * len(abnormalinds)):])

train_index = np.concatenate((train_normal, train_abnormal))
val_index = np.concatenate((val_normal, val_abnormal))
test_index = np.concatenate((test_normal, test_abnormal))

random.shuffle(train_index)
random.shuffle(val_index)
random.shuffle(test_index)

print("Train size: {}, normal size: {}, abnormal size: {}".format(len(train_index), len(train_normal), len(train_abnormal)))
print("Val size: {}, normal size: {}, abnormal size: {}".format(len(val_index), len(val_normal), len(val_abnormal)))
print("Test size: {}, normal size: {}, abnormal size: {}".format(len(test_index), len(test_normal), len(test_abnormal)))

print("Total size: {}, generate size: {}".format(len(labels), len(train_index) + len(val_index) + len(test_index)))

train_path = os.path.join(datadir, os.path.join(data, data + '_train.txt'))
val_path = os.path.join(datadir, os.path.join(data, data + '_val.txt'))
test_path = os.path.join(datadir, os.path.join(data, data + '_test.txt'))

np.savetxt(train_path, train_index, fmt='%d')
np.savetxt(val_path, val_index, fmt='%d')
np.savetxt(test_path, test_index, fmt='%d')

e = time.time()

print("Generate successfully, time cost: {:.2f}".format(e - s))
