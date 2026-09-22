import time
import random
import numpy as np
import os
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report, average_precision_score, precision_recall_curve, auc
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    if seed == 0:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

def load_data(datadir, data, khigh, klow, aug):
    print("Loading data.....")
    s = time.time()
    path = os.path.join(datadir, data)
    train_path = os.path.join(path, data + '_train.txt')
    train_index = np.loadtxt(train_path, dtype=np.int64)

    val_path = os.path.join(path, data + '_val.txt')
    val_index = np.loadtxt(val_path, dtype=np.int64)

    test_path = os.path.join(path, data + '_test.txt')
    test_index = np.loadtxt(test_path, dtype=np.int64)
    
    temp = TUDataset(root=datadir, name=data)
    if aug:
        LEs = torch.load(datadir + '/' + data + '/' + 'LM' + str(khigh) + '/Es.pt')
        LUs = torch.load(datadir + '/' + data + '/' + 'LM' + str(khigh) + '/Us.pt')

        SEs = torch.load(datadir + '/' + data + '/' + 'SM' + str(khigh) + '/Es.pt')
        SUs = torch.load(datadir + '/' + data + '/' + 'SM' + str(khigh) + '/Us.pt')

    dataset = []
    index = 0
    
    reducedx = 0
    if data == 'DBLP_v1' or data == 'TWITTER-Real-Graph-Partial':
        x = torch.load(datadir + '/' + data + '/' + 'reduced16' + '/reduced_x.pt')
        reducedx = 1

    for i, data in enumerate(temp):
        if aug:
            data['LEs'] = LEs[i].unsqueeze(0)
            data['LUs'] = LUs[i].unsqueeze(0)
            data['SEs'] = SEs[i].unsqueeze(0)
            data['SUs'] = SUs[i].unsqueeze(0)
            data['nodenum'] = torch.LongTensor([data.x.shape[0]])
        lap = get_laplacian(data.edge_index, normalization='sym')
        lap = to_dense_adj(edge_index=lap[0], edge_attr=lap[1], max_num_nodes=len(data.x))[0]
        adj = (torch.eye(len(lap)) - 0.5 * lap).to_sparse()
        data['edge_index'] = adj.indices()
        data['edge_attr'] = adj.values()
        data['plabel'] = data.y

        if reducedx:
            num_nodes = data.x.shape[0]
            data['x'] = x[index:index + num_nodes]  
            index += num_nodes
        
        dataset.append(data.cpu())

    trainset = [dataset[i] for i in train_index]
    valset = [dataset[i] for i in val_index]
    testset = [dataset[i] for i in test_index]
    unlabel_index = np.concatenate((val_index, test_index))
    unlabelset = [dataset[i] for i in unlabel_index]

    e = time.time()
    print("Loading successfully, train/val/test/unlabel size: {}/{}/{}/{}, time cost: {:.2f}".format(len(trainset), len(valset), len(testset), len(unlabelset), e - s))
    return dataset, trainset, valset, testset, unlabelset

def normal_abnormal_count(trainset):
    normalcount = 0
    abnormalcount = 0
    for data in trainset:
        if int(data.y) == 0:
            normalcount += 1
        elif int(data.y) == 1:
            abnormalcount += 1
    return normalcount, abnormalcount

def metrics(preds, labels):
    AUROC = roc_auc_score(labels, preds)
    precision, recall, _ = precision_recall_curve(labels, preds)
    AUPRC = auc(recall, precision)
    MF1 = f1_score(labels, preds, average='macro') 
    return AUROC, AUPRC, MF1

