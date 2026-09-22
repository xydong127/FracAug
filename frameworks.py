import torch.optim as optim
import torch.nn as nn
import torch 
from torch_geometric.utils import get_laplacian, to_dense_adj
import numpy as np
import torch.nn.functional as F
import time
from copy import deepcopy
from torch_geometric.loader import DataLoader
from sklearn import svm
from sklearn.metrics import hinge_loss
import random
from tqdm import tqdm
import os

import utils
import gnns

class base_framework(object):
    def __init__(self, device, baseline_config, info):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.best_AUROC = 0
        self.best_AUPRC = 0
        self.best_MF1 = 0
        self.best_epoch = 0
        self.best_model = None
        self.device = device
        self.baseline_config = baseline_config
        self.info = info

    def add_origin_train_loader(self, origin_train_loader):
        self.origin_train_loader = origin_train_loader

    def output(self, data_dict, model):
        batch = data_dict['batch']
        x = batch.x
        if "edge_index" in data_dict:
            edge_index = torch.cat(data_dict["edge_index"], dim=-1)
        else:
            edge_index = batch.edge_index
        if "edge_attr" in data_dict:
            edge_attr = torch.cat(data_dict["edge_attr"])
        else:
            edge_attr = batch.edge_attr
        batchind = batch.batch
        logit_embeds, node_embeds, graph_embeds = model(x, edge_index, edge_attr, batchind)
        return logit_embeds, node_embeds, graph_embeds

    def cal_loss(self, pred, truth):
        loss = self.criterion(pred, truth)
        return loss

    def train(self, train_loader):
        self.model.train()
        train_loss = 0
        if self.scheduler is not None:
            self.scheduler.step()
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch.to(self.device)
            data_dict = {"batch": batch}
            logit_embeds, node_embeds, graph_embeds = self.output(data_dict, self.model)
            loss = self.cal_loss(logit_embeds, batch.plabel)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        return train_loss / len(train_loader)

    def val(self, val_loader, epoch):
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch.to(self.device)
                data_dict = {"batch": batch}
                logit_embeds, node_embeds, graph_embeds = self.output(data_dict, self.model)
                logits = nn.functional.softmax(logit_embeds, dim=1)
                preds.append(logits)
                labels.append(batch.y)
            preds = torch.cat(preds, dim=0).detach().cpu().numpy().argmax(axis=1)
            labels = torch.cat(labels, dim=0).detach().cpu().numpy()
            val_AUROC, val_AUPRC, val_MF1 = utils.metrics(preds, labels)

        if self.best_AUROC + self.best_AUPRC + self.best_MF1 <= val_AUROC + val_AUPRC + val_MF1:
            self.best_AUROC = val_AUROC
            self.best_AUPRC = val_AUPRC
            self.best_MF1 = val_MF1
            self.best_epoch = epoch
            self.best_model = deepcopy(self.model)

        return val_AUROC, val_AUPRC, val_MF1

    def test(self, test_loader):
        self.best_model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in test_loader:
                batch.to(self.device)
                data_dict = {"batch": batch}
                logit_embeds, node_embeds, graph_embeds = self.output(data_dict, self.best_model)
                logits = nn.functional.softmax(logit_embeds, dim=1)
                preds.append(logits)
                labels.append(batch.y)
            preds = torch.cat(preds, dim=0).detach().cpu().numpy().argmax(axis=1)
            labels = torch.cat(labels, dim=0).detach().cpu().numpy()
            test_AUROC, test_AUPRC, test_MF1 = utils.metrics(preds, labels)
        return test_AUROC, test_AUPRC, test_MF1


class GNN_framework(base_framework):
    def __init__(self, device, baseline_config, info):
        super().__init__(device, baseline_config, info)
        self.model = gnns.GNN(baseline_config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=baseline_config['lr'], weight_decay=baseline_config['decay'])
        if baseline_config['baseline'] == 'GIN':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        normalcount = info['normal']
        abnormalcount = info['abnormal']
        total = normalcount + abnormalcount
        weight = torch.Tensor([total / normalcount, total / abnormalcount]).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.best_model = deepcopy(self.model)
        


class iGAD_framework(base_framework):
    def __init__(self, device, baseline_config, info):
        super().__init__(device, baseline_config, info)
        self.model = gnns.iGAD(baseline_config, self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=baseline_config['lr'])
        normalcount = info['normal']
        abnormalcount = info['abnormal']
        self.pred_prior = torch.log(torch.tensor([normalcount+1e-8, abnormalcount+1e-8], requires_grad=False)).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.best_model = deepcopy(self.model)

    def output(self, data_dict, model):
        batch = data_dict['batch']
        x = batch.x
        if "edge_index" in data_dict:
            edge_index = torch.cat(data_dict["edge_index"], dim=-1)
        else:
            edge_index = batch.edge_index
        if "edge_attr" in data_dict:
            edge_attr = torch.cat(data_dict["edge_attr"])
        else:
            edge_attr = batch.edge_attr
        adj = torch.sparse.FloatTensor(edge_index, edge_attr, torch.Size([len(x), len(x)]))
        batchind = batch.batch
        logit_embeds, node_embeds, graph_embeds = model(x, adj, batchind)
        return logit_embeds, node_embeds, graph_embeds

    def cal_loss(self, pred, truth):
        loss = self.criterion(pred + self.pred_prior, truth)
        return loss

class NS_framework(base_framework):
    def __init__(self, device, baseline_config, info):
        super().__init__(device, baseline_config, info)
        self.model = gnns.NS(baseline_config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=baseline_config['lr'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        normalcount = info['normal']
        abnormalcount = info['abnormal']
        total = normalcount + abnormalcount
        weight = torch.Tensor([total / normalcount, total / abnormalcount]).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.best_model = deepcopy(self.model)

class GLA_framework(base_framework):
    def __init__(self, device, baseline_config, info):
        super().__init__(device, baseline_config, info)
        self.model = gnns.GLA(baseline_config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=baseline_config['lr'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        normalcount = info['normal']
        abnormalcount = info['abnormal']
        total = normalcount + abnormalcount
        weight = torch.Tensor([total / normalcount, total / abnormalcount]).to(device)
        self.criterion = nn.NLLLoss(weight=weight)
        self.best_model = deepcopy(self.model)

    def get_one_hot_encoding(self, data, n_class):
        y = data.y.view(-1)
        encoding = np.zeros([len(y), n_class])
        for i in range(len(y)):
            encoding[i, int(y[i])] = 1
        return torch.from_numpy(encoding).to(self.device)
    
    def cal_loss(self, x1, x2):
        T = 0.5
        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim 
        loss = - torch.log(loss).mean()

        return loss

    def train(self, train_loader):
        
        dataset = train_loader.dataset
        random.shuffle(dataset)
        dataset1 = dataset
        dataset2 = deepcopy(dataset1)
        batch_size = self.baseline_config['batchsize']

        loader1 = DataLoader(dataset1, batch_size, shuffle=False)
        loader2 = DataLoader(dataset2, batch_size, shuffle=False)

        self.model.train()
        total_loss = 0
        for data1, data2 in zip(loader1, loader2):
            self.optimizer.zero_grad()
            data1 = data1.to(self.device)
            data2 = data2.to(self.device)

            out1, x1, pred1, pred_gcn1 = self.model.forward_cl(data1.x, data1.edge_index, data1.edge_attr, data1.batch)
            out2, _, pred2, pred_gcn2 = self.model.forward_cl(data2.x, data2.edge_index, data2.edge_attr, data2.batch)

            eq = torch.argmax(pred1, axis=-1) - torch.argmax(pred2, axis=-1)
            indices = (eq == 0).nonzero().reshape(-1)
            loss = self.cal_loss(out1[indices], out2[indices])
            loss += (self.criterion(pred1, data1.plabel.view(-1)) + self.criterion(pred2[indices], data2.plabel.view(-1)[indices]))

            if len(indices) > 1:
                loss.backward()
                total_loss += loss.item() * data1.num_graphs
                self.optimizer.step()
        return total_loss / len(loader1.dataset)

class GMixup_framework(base_framework):
    def __init__(self, device, baseline_config, info):
        super().__init__(device, baseline_config, info)
        self.model = gnns.GMixup(baseline_config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=baseline_config['lr'], weight_decay=baseline_config['decay'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        normalcount = info['normal']
        abnormalcount = info['abnormal']
        total = normalcount + abnormalcount
        self.weight = torch.Tensor([total / normalcount, total / abnormalcount]).to(device)
        self.best_model = deepcopy(self.model)

    def cal_loss(self, pred, truth):
        pred = F.log_softmax(pred, dim=-1)
        truth = F.one_hot(truth,2)
        loss = - torch.sum(pred * truth * self.weight)
        return loss / pred.size()[0]

class FGWMixup_framework(base_framework):
    def __init__(self, device, baseline_config, info):
        super().__init__(device, baseline_config, info)
        self.model = gnns.FGWMixup(baseline_config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=baseline_config['lr'], weight_decay=baseline_config['decay'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        normalcount = info['normal']
        abnormalcount = info['abnormal']
        total = normalcount + abnormalcount
        self.weight = torch.Tensor([total / normalcount, total / abnormalcount]).to(device)
        self.best_model = deepcopy(self.model)

    def cal_loss(self, pred, truth):
        pred = F.log_softmax(pred, dim=-1)
        truth = F.one_hot(truth,2)
        loss = - torch.sum(pred * truth * self.weight)
        return loss / pred.size()[0]

class LRGNN_framework(base_framework):
    def __init__(self, device, baseline_config, info):
        super().__init__(device, baseline_config, info)
        self.model = gnns.LRGNN(baseline_config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=baseline_config['lr'], weight_decay=baseline_config['decay'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(100), eta_min=0)
        normalcount = info['normal']
        abnormalcount = info['abnormal']
        total = normalcount + abnormalcount
        weight = torch.Tensor([total / normalcount, total / abnormalcount]).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.best_model = deepcopy(self.model)

class GmapAD_framework(base_framework):
    def __init__(self, device, baseline_config, info):
        super().__init__(device, baseline_config, info)
        self.model = gnns.GmapAD(baseline_config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=baseline_config['lr'])
        self.criterion = nn.CrossEntropyLoss()
        self.best_model = deepcopy(self.model)
    
    def cal_loss(self, pred, truth):
        pred = F.softmax(pred, dim=1)
        mask_0 = [False for _ in range(truth.shape[0])]
        mask_1 = [False for _ in range(truth.shape[0])]
        for i, l in enumerate(truth):
            if l == 1:
                mask_1[i] = True
                mask_0[i] = False
            else:
                mask_1[i] = False
                mask_0[i] = True

        loss = self.criterion(pred[mask_0], truth[mask_0]) + self.criterion(pred[mask_1], truth[mask_1])
        return loss
    
    def gen_can_pool(self, W, node_pool):
        pool_candidates = []
        for i, w in enumerate(W):
            w = w.view(-1,1)
            candidate = torch.mul(node_pool, w)
            pool_candidates.append(candidate)

        return pool_candidates

    def mutation_cross_w(self, W):

        mut_rate = self.baseline_config['mut_rate'] 
        cros_rate = self.baseline_config['cros_rate']
    
        new_cands = []
    
        for i, candidate in enumerate(W):
            r = random.sample(range(len(W)),5)
            while i in r:
                r = random.sample(range(len(W)),5)
            mutated_cand = W[r[0]] + mut_rate * (W[r[1]] + W[r[2]]) + mut_rate * (W[r[3]] + W[r[4]])
            zero = torch.zeros_like(mutated_cand)
            one = torch.ones_like(mutated_cand)
            mutated_cand = torch.where(mutated_cand>=2, one, mutated_cand)
            mutated_cand = torch.where(mutated_cand<2, zero, mutated_cand)

            cros_cand = mutated_cand
            for j, vij in enumerate(mutated_cand):
                pos = torch.rand(1)
                if pos < cros_rate or (i==j):
                    break
                else:
                    cros_cand[j] = candidate[j]
            new_cands.append(cros_cand)
        new_cands = torch.stack(new_cands).view(W.shape)
    
        return new_cands

    def evo_classify(self, clf, X_train, Y_train):

        clf.fit(X_train, Y_train)
        x_train_pred = clf.predict(X_train)
        svm_loss = hinge_loss(Y_train, x_train_pred)
        return svm_loss

    def test(self, test_loader):
        pos_graphs = []
        for batch in self.origin_train_loader:
            for i in range(len(batch.y)):
                if int(batch.y[i]) == 1:
                    pos_graphs.append(batch[i])
        pos_loader = DataLoader(pos_graphs, batch_size=self.baseline_config['batchsize'], shuffle=True)

        node_embeds_list = []
        graph_embeds_list = []
        for batch in pos_loader: 
            batch.to(self.device)
            logit_embeds, node_embeds, graph_embeds = self.best_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            node_embeds_list.append(node_embeds)
            graph_embeds_list.append(graph_embeds)

        node_embeds = torch.cat(node_embeds_list)
        graph_embeds = torch.cat(graph_embeds_list)

        node_embeds = F.normalize(node_embeds, p=2, dim=-1)
        graph_embeds = F.normalize(graph_embeds, p=2, dim=-1)
        cos_sim = torch.matmul(graph_embeds, node_embeds.T)
        node_sims = cos_sim.sum(dim=0)
        top_k_nodes = torch.topk(node_sims, self.baseline_config['k'], -1, True).indices
        node_pool = node_embeds.index_select(0, top_k_nodes).detach().cpu()
        
        clf = svm.SVC(kernel='linear', C=1.0, cache_size=1000, class_weight='balanced')
        
        train_graph_embeds_list = []
        train_graph_labels_list = []
        
        for batch in self.origin_train_loader:
            batch.to(self.device)
            logit_embeds, node_embeds, graph_embeds = self.best_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            train_graph_embeds_list.append(graph_embeds.detach().cpu())
            train_graph_labels_list.append(batch.y.detach().cpu())
        
        train_graph_embeds = torch.cat(train_graph_embeds_list)
        Y_train = torch.cat(train_graph_labels_list).numpy()
        
        
        test_graph_embeds_list = []
        test_graph_labels_list = []
        for batch in test_loader:
            batch.to(self.device)
            logit_embeds, node_embeds, graph_embeds = self.best_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            test_graph_embeds_list.append(graph_embeds.detach().cpu())
            test_graph_labels_list.append(batch.y.detach().cpu())

        test_graph_embeds = torch.cat(test_graph_embeds_list)
        labels = torch.cat(test_graph_labels_list).numpy()


        old_W = []
        num_nodes = node_pool.shape[0]
        cand_size = self.baseline_config['cand_size']
        w = torch.ones(num_nodes, 1, requires_grad=False)
        old_W.append(w)
        for i in range(cand_size-1):
            n = range(1, num_nodes)
            k = random.randint(0, num_nodes-1)
            ones = random.sample(n, k)
            w = torch.zeros(num_nodes, 1, requires_grad=False)
            w[ones] = 1
            old_W.append(w)
        old_W = torch.stack(old_W).view(cand_size, num_nodes)
        
        ini_cands = self.gen_can_pool(old_W, node_pool)
        
        best_svm_losses = []
        
        for i, cand in enumerate(ini_cands):
            h_gs = torch.cdist(train_graph_embeds, cand, p=1)
            X_train = h_gs.cpu().numpy()
            
            svm_loss = self.evo_classify(clf, X_train, Y_train)
            best_svm_losses.append(svm_loss)
        
        evo_tolerance = 10
        cur_tolerance = 0
        global_min_loss = 100

        for epoch in range(self.baseline_config['evo_nepoch']):
            new_W = self.mutation_cross_w(old_W)
            new_cands = self.gen_can_pool(new_W, node_pool)

            for i, cand in enumerate(new_cands):
                h_gs = torch.cdist(train_graph_embeds, cand, p=1)
                X_train = h_gs.cpu().numpy()

                svm_loss = self.evo_classify(clf, X_train, Y_train)

                if best_svm_losses[i] < svm_loss:
                     new_W[i] = old_W[i]
                else:
                    best_svm_losses[i] = svm_loss

            old_W = new_W
            min_loss = min(best_svm_losses)
            if min_loss < global_min_loss:
                global_min_loss= min_loss
            else:
                cur_tolerance = cur_tolerance + 1
            if cur_tolerance > evo_tolerance:
                break
        
        min_svm_loss = min(best_svm_losses)

        best_w = old_W[best_svm_losses.index(min_svm_loss)]
        best_w = best_w.view(-1,1)
        best_cands = torch.mul(node_pool, best_w)
        
        h_gs = torch.cdist(train_graph_embeds, best_cands, p=1) 
        X_train = h_gs.cpu().numpy()

        h_gs = torch.cdist(test_graph_embeds, best_cands, p=1)
        X_test = h_gs.cpu().numpy()
        
        clf.fit(X_train, Y_train)
        preds = clf.predict(X_test)
        
        test_AUROC, test_AUPRC, test_MF1 = utils.metrics(preds, labels)
        return test_AUROC, test_AUPRC, test_MF1

class GRDL_framework(base_framework):
    def __init__(self, device, baseline_config, info):
        super().__init__(device, baseline_config, info)
        self.model = gnns.GRDL(baseline_config, info, self.device).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.model.extractor.parameters()},
            {'params': self.model.mmd.atoms},
            {'params': self.model.mmd.gamma, 'lr': baseline_config['lr2']}
        ], lr=baseline_config['lr1'])
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        normalcount = info['normal']
        abnormalcount = info['abnormal']
        total = normalcount + abnormalcount
        self.weight = torch.Tensor([total / normalcount, total / abnormalcount]).to(device)
        self.best_model = deepcopy(self.model)
        self.lam = baseline_config['lam']

    def cal_loss(self, pred, truth):
        loss1 = torch.sum(F.one_hot(truth, 2) * (-torch.log(pred)*self.weight), dim=1).mean()
        loss2 = self.model.mmd.discriminate_loss()
        loss = loss1 + self.lam * loss2
        return loss

class RQGNN_framework(base_framework):
    def __init__(self, device, baseline_config, info):
        super().__init__(device, baseline_config, info)
        self.model = gnns.RQGNN(baseline_config, self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=baseline_config['lr'])
        self.normalcount = info['normal']
        self.abnormalcount = info['abnormal']
        self.beta = baseline_config['beta']
        self.gamma = baseline_config['gamma']
        self.best_model = deepcopy(self.model)

    def output(self, data_dict, model):
        batch = data_dict['batch']
        xlxs = []
        node_belongs = []
        idx = 0
        for i in range(len(batch.y)):
            x = batch[i].x
            if 'edge_index' in data_dict and 'edge_attr' in data_dict:
                lap = get_laplacian(data_dict['edge_index'][i] - idx, data_dict['edge_attr'][i], normalization='sym')
            else:
                lap = get_laplacian(batch[i].edge_index, batch[i].edge_attr, normalization='sym')
            lap = to_dense_adj(edge_index=lap[0], edge_attr=lap[1], max_num_nodes=len(x))[0]
            xlx = torch.diag(torch.mm(torch.mm(x.T, lap), x)).unsqueeze(0)         
            xlxs.append(xlx)
            node_belongs.append(list(range(idx, idx + len(x))))
            idx += len(x)
            
        xlxs = torch.cat(xlxs)

        x = batch.x
        if "edge_index" in data_dict:
            edge_index = torch.cat(data_dict["edge_index"], dim=-1)
        else:
            edge_index = batch.edge_index
        if "edge_attr" in data_dict:
            edge_attr = torch.cat(data_dict["edge_attr"])
        else:
            edge_attr = batch.edge_attr
        batchind = batch.batch

        logit_embeds, node_embeds, graph_embeds = model(x, edge_index, edge_attr, batchind, xlxs, node_belongs)

        return logit_embeds, node_embeds, graph_embeds

    def focal_loss(self, labels, logits, alpha, gamma):
        BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss

    def cal_loss(self, pred, truth):
        samples_per_cls = [self.normalcount, self.abnormalcount]
        nclass = 2
        effective_num = 1.0 - np.power(self.beta, samples_per_cls)
        if effective_num[1] > 0:
            weights = (1.0 - self.beta) / np.array(effective_num)
        else:
            weights = np.array([(1.0 - self.beta) / effective_num[0], 0])
        weights = weights / np.sum(weights) * nclass

        labels_one_hot = F.one_hot(truth, nclass).float()

        weights = torch.tensor(weights).float().to(self.device)
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, nclass)

        loss = self.focal_loss(labels_one_hot, pred, weights, self.gamma)

        return loss


class UniGAD_framework(base_framework):
    def __init__(self, device, baseline_config, info):
        super().__init__(device, baseline_config, info)
        self.pretrain_model = gnns.UniGADGraphMAE(baseline_config, self.device).to(self.device)
        self.model = None
        self.optimizer = None
        normalcount = info['normal']
        abnormalcount = info['abnormal']
        total = normalcount + abnormalcount
        weight = torch.Tensor([total / normalcount, total / abnormalcount]).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.best_model = deepcopy(self.model)

    def pretrain(self, data, dataset_loader):
        self.pretrain_model.train()
        print("Start pretraining......")
        model_path = 'pretrain_models/UniGAD_{}_pretrain_model.pt'.format(data)
        if os.path.exists(model_path):
            self.pretrain_model.load_state_dict(torch.load(model_path))
        else:
            optimizer = optim.Adam(self.pretrain_model.parameters(), lr=1e-2)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)
            max_epoch = 100
            epoch_iter = tqdm(range(max_epoch))

            for epoch in epoch_iter:
                for batch in dataset_loader:
                    batchgpu = batch.to(self.device)
                    x = batchgpu.x
                    edge_index = batchgpu.edge_index
                    edge_attr = batchgpu.edge_attr
                    loss = self.pretrain_model(x, edge_index, edge_attr)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    scheduler.step()
            
            os.makedirs('pretrain_models', exist_ok=True)
            torch.save(self.pretrain_model.state_dict(), model_path)

        print("Pretraining successfully")

        self.pretrain_model.eval()
        self.model = gnns.UniGADMLP_E2E(self.pretrain_model, self.baseline_config, self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.baseline_config['lr'])
