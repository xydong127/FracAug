import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch_geometric.loader import DataLoader
import torch.optim as optim
from copy import deepcopy

import utils

class AUGGAD(nn.Module):
    def __init__(self, thigh, tlow, eps):
        super(AUGGAD, self).__init__()
        self.thigh = thigh
        self.tlow = tlow
        self.eps = eps
        lr1 = 0
        lr2 = 3
        self.Lpowers = nn.Parameter((lr1 - lr2) * torch.rand(thigh) + lr2)
        self.Lalphas = nn.Parameter(torch.rand(thigh))
        sr1 = 0
        sr2 = 3
        self.Spowers = nn.Parameter((sr1 - sr2) * torch.rand(tlow) + sr2)
        self.Salphas = nn.Parameter(torch.rand(tlow))
        self.alphas = nn.Parameter(torch.rand(2))

    def forward(self, data):
        LEs, LUs, SEs, SUs, nodenum = data.LEs, data.LUs, data.SEs, data.SUs, data.nodenum
        
        LM = 0
        Lalphas = F.softmax(self.Lalphas, dim=0)
        Lpowers = F.relu(self.Lpowers)
        for t in range(self.thigh):
            LM += Lalphas[t] * torch.bmm(torch.bmm(LUs, torch.pow(LEs, Lpowers[t])), torch.transpose(LUs, 1, 2))

        SM = 0
        Salphas = F.softmax(self.Salphas, dim=0)
        Spowers = F.relu(self.Spowers)
        for t in range(self.tlow):
            SM += Salphas[t] * torch.bmm(torch.bmm(SUs, torch.pow(SEs, Spowers[t])), torch.transpose(SUs, 1, 2))

        alphas = F.softmax(self.alphas, dim=0)
        M = alphas[0] * LM + alphas[1] * SM

        edge_index = []
        edge_weight = []
        M[M <= self.eps] = 0

        count = 0
        
        for i, num in enumerate(nodenum):
            temp = M[i][:num, :num].to_sparse()
            edge_index.append(temp.indices() + count)
            edge_weight.append(temp.values())
            count += num

        return edge_index, edge_weight

def build_aug(thigh, tlow, eps, device, lr, info):
    auggad = AUGGAD(thigh, tlow, eps).to(device)
    aug_optimizer = optim.Adam(auggad.parameters(), lr=lr)
    aug_cos = nn.CosineSimilarity()
    normalcount = info['normal']
    abnormalcount = info['abnormal']
    total = normalcount + abnormalcount
    weight = torch.Tensor([total / normalcount, total / abnormalcount]).to(device)
    aug_entropy = nn.CrossEntropyLoss(weight=weight)
    return auggad, aug_optimizer, aug_cos, aug_entropy

def train_aug(framework, auggad, augnepoch, train_loader, val_loader, aug_optimizer, aug_cos, aug_entropy, device):
    framework.best_model.eval()
    bestauggad = auggad
    best_AUROC, best_AUPRC, best_MF1, best_epoch = 0, 0, 0, 0
    for augepoch in range(augnepoch):
        auggad.train()
        aug_train_start = time.time()
        aug_loss = 0
        for batch in train_loader:
            aug_optimizer.zero_grad()
            batch.to(device)
            aug_edge_index, aug_edge_weight = auggad(batch)
            data_dict = {"batch": batch}
            logit_embeds, node_embeds, graph_embeds = framework.output(data_dict, framework.best_model)
            data_dict["edge_index"] = aug_edge_index
            data_dict["edge_attr"] = aug_edge_weight
            aug_logit_embeds, node_embeds, aug_graph_embeds = framework.output(data_dict, framework.best_model)
            aug_graph_embeds = torch.empty_like(aug_graph_embeds).copy_(aug_graph_embeds)
            distances = ((1 - aug_cos(aug_graph_embeds, graph_embeds)) / 2).unsqueeze(-1) * F.one_hot(batch.plabel)
            aug_logit_embeds = aug_logit_embeds - distances

            loss = aug_entropy(aug_logit_embeds, batch.plabel)
            loss.backward()
            aug_optimizer.step()
            aug_loss += loss.item()

        aug_train_end = time.time()
        
        auggad.eval()
        aug_val_start = time.time()
        preds = []
        labels = []
        for batch in val_loader:
            batch.to(device)
            aug_edge_index, aug_edge_weight = auggad(batch)
            data_dict = {"batch": batch}
            data_dict["edge_index"] = aug_edge_index
            data_dict["edge_attr"] = aug_edge_weight
            aug_logit_embeds, node_embeds, aug_graph_embeds = framework.output(data_dict, framework.best_model)
            aug_logit_probs = nn.functional.softmax(aug_logit_embeds, dim=1)

            preds.append(aug_logit_probs)
            labels.append(batch.plabel)
        preds = torch.cat(preds, dim=0).detach().cpu().numpy().argmax(axis=1)
        labels = torch.cat(labels, dim=0).detach().cpu().numpy()
        val_AUROC, val_AUPRC, val_MF1 = utils.metrics(preds, labels)

        if best_AUROC + best_AUPRC + best_MF1 <= val_AUROC + val_AUPRC + val_MF1:
            best_AUROC = val_AUROC
            best_AUPRC = val_AUPRC
            best_MF1 = val_MF1
            best_epoch = augepoch
            bestauggad = deepcopy(auggad)
        aug_val_end = time.time()

    return bestauggad


def gen_data(framework, auggad, trainset, unlabelset, unlabel_loader, normal, abnormal, batchsize, times, device):
    framework.best_model.eval()
    auggad.eval()
    new_trainset = []
    new_trainset.extend(trainset)
    pseudo_start = time.time()
    logits = []
    aug_logits = []
    with torch.no_grad():
        for batch in unlabel_loader:
            batch.to(device)
            aug_edge_index, aug_edge_weight = auggad(batch)
            data_dict = {"batch": batch}
            logit_embeds, node_embeds, graph_embeds = framework.output(data_dict, framework.best_model)
            data_dict["edge_index"] = aug_edge_index
            data_dict["edge_attr"] = aug_edge_weight
            aug_logit_embeds, node_embeds, aug_graph_embeds = framework.output(data_dict, framework.best_model)
            logit_embeds = nn.functional.softmax(logit_embeds, dim=1)
            aug_logit_embeds = nn.functional.softmax(aug_logit_embeds, dim=1)
    
            logit1 = logit_embeds[:, 1]
            aug_logit1 = aug_logit_embeds[:, 1]
            
            logits.append(logit1.T.squeeze(0))
            aug_logits.append(aug_logit1.T.squeeze(0))
        
        logits = torch.cat(logits)
        aug_logits = torch.cat(aug_logits)


        pseudo_normal = (torch.where(logits < normal, 1, 0) & torch.where(aug_logits < normal, 1, 0)).nonzero().T.squeeze(0)
        pseudo_abnormal = (torch.where(logits > abnormal, 1, 0) & torch.where(aug_logits > abnormal, 1, 0)).nonzero().T.squeeze(0)
        
        normal_aug_logits = aug_logits[pseudo_normal]
        abnormal_aug_logits = aug_logits[pseudo_abnormal]
        
        num_gen_normal = int(framework.info['normal'] *  times)
        num_gen_abnormal = int(framework.info['abnormal'] * times)
        
        if len(normal_aug_logits) > num_gen_normal:
            _, normal_ids = torch.topk(normal_aug_logits, k=num_gen_normal, largest=False)
            pseudo_normal = pseudo_normal[normal_ids]
        if len(abnormal_aug_logits) > num_gen_abnormal:
            _, abnormal_ids = torch.topk(abnormal_aug_logits, k=num_gen_abnormal, largest=True)
            pseudo_abnormal = pseudo_abnormal[abnormal_ids]


        pseudo_normal = pseudo_normal.tolist()
        pseudo_abnormal = pseudo_abnormal.tolist()

        pseudo_normal = [unlabelset[i] for i in pseudo_normal]
        for i in range(len(pseudo_normal)):
            pseudo_normal[i].plabel = torch.zeros(1).long()
        pseudo_abnormal = [unlabelset[i] for i in pseudo_abnormal]
        for i in range(len(pseudo_abnormal)):
            pseudo_abnormal[i].plabel = torch.ones(1).long()
        
        gennormal = len(pseudo_normal)
        genabnormal = len(pseudo_abnormal)
        new_trainset.extend(pseudo_normal)
        new_trainset.extend(pseudo_abnormal)

    pseudo_end = time.time()
    return new_trainset

