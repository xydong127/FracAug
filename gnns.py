import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv, ChebConv, JumpingKnowledge, LayerNorm, BatchNorm
from torch.nn.parameter import Parameter
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
from functools import partial
from sklearn.cluster import KMeans
from torch_geometric.loader import DataLoader
import numpy as np

### GNN 
class GNN(nn.Module):
    def __init__(self, baseline_config):
        super(GNN, self).__init__()
        in_dim = baseline_config['in_dim']
        hid_dim = baseline_config['hid_dim']
        out_dim = baseline_config['out_dim']
        layer_num = baseline_config['layer_num']
        dropout = baseline_config['dropout']
        baseline = baseline_config['baseline']
        
        if baseline == 'GCN':
            self.conv = GCN(in_channels=in_dim, hidden_channels=hid_dim, num_layers=layer_num, out_channels=hid_dim, dropout=dropout)
        elif baseline == 'SAGE':
            self.conv = GraphSAGE(in_channels=in_dim, hidden_channels=hid_dim, num_layers=layer_num, out_channels=hid_dim, dropout=dropout, norm=nn.BatchNorm1d(hid_dim))
        elif baseline == 'GAT':
            self.conv = GAT(in_channels=in_dim, hidden_channels=hid_dim, num_layers=layer_num, out_channels=hid_dim, dropout=dropout, heads=baseline_config['heads'], act='elu', norm=nn.BatchNorm1d(hid_dim))
        elif baseline == 'GIN':
            self.conv = GIN(in_channels=in_dim, hidden_channels=hid_dim, num_layers=layer_num, out_channels=hid_dim, dropout=dropout, eps=baseline_config['eps'], train_eps=True)
        else:
            print("Error model: {}".format(baseline))
            exit()
        
        self.nn = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))
        self.bn = nn.BatchNorm1d(hid_dim)

        self.out = nn.Linear(hid_dim, out_dim)
        if baseline == 'GAT' or baseline == 'SAGE':
            self.pool = global_add_pool
        else:
            self.pool = global_mean_pool

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batchind):
        node_embeds = self.conv(x=x, edge_index=edge_index, edge_weight=edge_attr)
        node_embeds = self.nn(node_embeds)
        node_embeds = self.bn(node_embeds)
        graph_embeds = self.pool(node_embeds, batchind)
        logit_embeds = self.out(graph_embeds)
        return logit_embeds, None, graph_embeds

### iGAD
class iGAD_graph_convolution_layer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, device, bias=False):
        super(iGAD_graph_convolution_layer, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.device = device
        self.weight = Parameter(torch.FloatTensor(in_features, hidden_features))
        self.weight2 = Parameter(torch.FloatTensor(hidden_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(bias)
        self.mlp_layer_1 = nn.Linear(self.in_features, self.hidden_features, bias=True)
        self.mlp_layer_2 = nn.Linear(self.hidden_features, self.out_features, bias=True)
        self.relu = nn.ReLU()

    def reset_parameters(self, bias):

        nn.init.kaiming_normal_(self.weight)
        nn.init.kaiming_normal_(self.weight2)
        if bias:
            self.bias.data.uniform_(-1, 1)

    def forward(self, x, adj):
        conv_layer_1_output = self.relu(torch.mm(torch.sparse.mm(adj, x), self.weight))
        conv_layer_2_output = torch.mm(torch.sparse.mm(adj, conv_layer_1_output), self.weight2)
        self_contribution_layer_output = self.mlp_layer_2(self.relu(self.mlp_layer_1(x)))
        outputs = torch.cat((self_contribution_layer_output, conv_layer_1_output, conv_layer_2_output), dim=1)

        return outputs

class iGAD_graph_convolution(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, device):

        super(iGAD_graph_convolution, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pool = global_mean_pool
        self.device = device
        self.gc = iGAD_graph_convolution_layer(self.input_dim, self.hidden_dim, self.output_dim, self.device)
        self.mlp_1 = nn.Linear(self.hidden_dim + self.output_dim*2, self.hidden_dim)
        self.mlp_2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, adj, batchind):

        h = self.gc(x, adj)
        h = self.mlp_2(F.relu(self.mlp_1(h)))
        H = self.pool(h, batchind)
        return H

class iGAD_RW_GNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_subgraphs, size_subgraph, max_step, normalize, dropout, device):
        super(iGAD_RW_GNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_subgraphs = n_subgraphs
        self.size_subgraphs = size_subgraph
        self.max_step = max_step
        self.device = device
        self.normalize = normalize
        self.Theta_matrix = Parameter(
            torch.FloatTensor(self.n_subgraphs, self.size_subgraphs * (self.size_subgraphs - 1) // 2, 1))
        self.bn = nn.BatchNorm1d(self.n_subgraphs * (self.max_step-1))
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()
        self.layer_1 = nn.Linear(self.n_subgraphs*(self.max_step-1), self.hidden_dim, bias= True)
        self.layer_2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        self.relu = nn.ReLU()

    def init_weights(self):
        nn.init.kaiming_normal_(self.Theta_matrix)

    def forward(self, adj, batchind):
        sampled_matrix = self.relu(self.Theta_matrix)
        sampled_matrix = sampled_matrix[:, :, 0]
        adj_sampled = torch.zeros(self.n_subgraphs, self.size_subgraphs, self.size_subgraphs).to(self.device)
        idx = torch.triu_indices(self.size_subgraphs, self.size_subgraphs, offset=1)
        adj_sampled[:, idx[0], idx[1]] = sampled_matrix
        adj_sampled = adj_sampled + torch.transpose(adj_sampled, 1, 2)
        unique, counts = torch.unique(batchind, return_counts=True)
        n_graphs = unique.size(0)
        n_nodes = adj.shape[0]

        if self.normalize:
            norm = counts.unsqueeze(1).repeat(1, self.n_subgraphs)

        E = torch.ones((self.n_subgraphs, self.size_subgraphs, n_nodes), device=self.device)

        I = torch.eye(n_nodes, device=self.device)
        adj_power = adj
        P_power_E = E
        random_walk_results = list()

        for i in range(1, self.max_step):
            I = torch.sparse.mm(adj_power, I)
            P_power_E = torch.einsum("abc,acd->abd", (adj_sampled, P_power_E))
            res = torch.einsum("abc,cd->abd", (P_power_E, I))
            res = torch.zeros(res.size(0), res.size(1), n_graphs, device=self.device).index_add_(2, batchind, res)
            res = torch.sum(res, dim=1)
            res = torch.transpose(res, 0, 1)
            if self.normalize:
                res /= norm
            random_walk_results.append(res)


        random_walk_results = torch.cat(random_walk_results, dim=1)
        random_walk_results = self.bn(random_walk_results)
        random_walk_results = self.layer_2(self.relu(self.layer_1(random_walk_results)))
        return random_walk_results

class iGAD(nn.Module):
    def __init__(self, baseline_config, device):
        super(iGAD, self).__init__()
        in_dim = baseline_config['in_dim']
        self.dim_1 = baseline_config['dim_1']
        f_hidden_dim = baseline_config['f_hidden_dim']
        f_output_dim = baseline_config['f_output_dim']
        t_hidden_dim = baseline_config['t_hidden_dim']
        t_output_dim = baseline_config['t_output_dim']
        out_dim = baseline_config['out_dim']
        self.n_subgraphs = baseline_config['n_subgraphs']
        self.size_subgraphs = baseline_config['size_subgraphs']
        self.max_step = baseline_config['max_step']
        self.normalize = baseline_config['normalize']
        self.dropout = baseline_config['dropout']
        self.device = device

        self.SEAG_features = iGAD_graph_convolution(in_dim, f_hidden_dim, f_output_dim, device)
        self.SEAG_topo = iGAD_RW_GNN(t_hidden_dim, t_output_dim, self.n_subgraphs, self.size_subgraphs, self.max_step, self.normalize, self.dropout, device)
        self.mlp_1 = nn.Linear(f_output_dim + t_output_dim, self.dim_1, bias=True)
        self.mlp_2 = nn.Linear(self.dim_1, out_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x, adj, batchind):

        outputs_1 = self.SEAG_features(x, adj, batchind)
        outputs_2 = self.SEAG_topo(adj, batchind)

        h = torch.cat((outputs_1, outputs_2), dim=1)
        graph_embeds = self.relu(self.mlp_1(h))
        logit_embeds = self.mlp_2(graph_embeds)

        return logit_embeds, None, graph_embeds

### NodeSam and SubMix
class NSMLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_units=32, num_layers=1):
        super(NSMLP, self).__init__()
        if num_layers == 1:
            self.layers = nn.Linear(num_features, num_classes)
        elif num_layers > 1:
            layers = [nn.Linear(num_features, hidden_units),
                      nn.BatchNorm1d(hidden_units),
                      nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_units, hidden_units),
                               nn.BatchNorm1d(hidden_units),
                               nn.ReLU()])
            layers.append(nn.Linear(hidden_units, num_classes))
            self.layers = nn.Sequential(*layers)
        else:
            raise ValueError()

    def forward(self, x):
        return self.layers(x)

class NS(nn.Module):
    def __init__(self, baseline_config):
        super(NS, self).__init__()
        num_features = baseline_config['in_dim']
        hidden_units = baseline_config['hid_dim']
        num_classes = baseline_config['out_dim']
        num_layers = baseline_config['layer_num']
        dropout = baseline_config['dropout']
        mlp_layers = baseline_config['mlp_layers']

        convs, bns = [], []
        mlps = []
        linears = []
        for i in range(num_layers):
            input_dim = num_features if i == 0 else hidden_units
            mlps.append(NSMLP(input_dim, hidden_units, hidden_units, mlp_layers))
            convs.append(GIN(in_channels=hidden_units, hidden_channels=hidden_units, out_channels=hidden_units, num_layers=1))
            bns.append(nn.BatchNorm1d(hidden_units))
            linears.append(nn.Linear(hidden_units, num_classes))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.mlps = nn.ModuleList(mlps)
        self.linears = nn.ModuleList(linears)
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        h_list = [x]
        count = 0
        for conv, bn in zip(self.convs, self.bns):
            h = self.mlps[count](h_list[-1])
            h_list.append(torch.relu(bn(h)))
            h = conv(h, edge_index, edge_weight=edge_attr)
            count += 1
        h_list.append(torch.relu(bn(h)))
        h_list = h_list[1:]
        logit_embeds = 0
        graph_embeds = 0
        
        for i in range(self.num_layers):
            h_pooled = global_add_pool(h_list[i], batch)
            graph_embeds += h_pooled
            h_pooled = self.linears[i](h_pooled)
            logit_embeds += F.dropout(h_pooled, self.dropout, self.training)
        return logit_embeds, None, graph_embeds

### GLA
class GLAConv(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 edge_norm=True,
                 gfn=False):
        super(GLAConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        loop_weight = torch.full((num_nodes, ),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        edge_index = edge_index[0]
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x

        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = GLAConv.norm(
                    edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class GLA(torch.nn.Module):
    def __init__(self, baseline_config):
        super(GLA, self).__init__()
        hidden_in = baseline_config['in_dim']
        num_feat_layers = baseline_config['num_feat_layers']
        self.conv_residual = baseline_config['residual']
        self.fc_residual = False 
        self.n_class = baseline_config['out_dim']
        self.global_pool = global_mean_pool
        self.dropout = baseline_config['dropout']
        hidden = baseline_config['hid_dim']
        num_conv_layers = baseline_config['num_conv_layers']
        num_fc_layers = baseline_config['num_fc_layers']
        edge_norm = baseline_config['edge_norm']
        gfn = False
        GConv = partial(GLAConv, edge_norm=edge_norm, gfn=gfn)
        self.bn_feat = nn.BatchNorm1d(hidden_in)
        feat_gfn = True
        self.conv_feat = GLAConv(hidden_in, hidden, gfn=feat_gfn)
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.bns_conv.append(nn.BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))
        self.bn_hidden = nn.BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for i in range(num_fc_layers - 1):
            self.bns_fc.append(nn.BatchNorm1d(hidden))
            self.lins.append(nn.Linear(hidden, hidden))
        self.lin_class = nn.Linear(hidden, self.n_class)
        self.lin_class2 = nn.Linear(hidden, self.n_class)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

        self.proj_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
        self.proj_head2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index, batch, edge_attr))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index, batch, edge_attr))
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)

        graph_embeds = x
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        logit_embeds = self.lin_class(x)
        return logit_embeds, None, graph_embeds

    def predictor(self, x):
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_cl(self, x, edge_index, edge_attr, batch):
        
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index, batch, edge_attr))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index, batch, edge_attr))
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)

        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        out = x
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        
        out = self.proj_head(out)
        pred = self.predictor(x)

        pred_gcn = pred
        return out, x, pred, pred_gcn

### GMixup
class GMixup(nn.Module):
    def __init__(self, baseline_config):
        super(GMixup, self).__init__()
        num_features = baseline_config['in_dim']
        dim = baseline_config['hid_dim']
        num_classes = baseline_config['out_dim']

        self.nn1 = nn.Sequential(nn.Linear(num_features, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv1 = GIN(in_channels=dim, hidden_channels=dim, out_channels=dim, num_layers=1)
        self.bn1 = nn.BatchNorm1d(dim)

        self.nn2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv2 = GIN(in_channels=dim, hidden_channels=dim, out_channels=dim, num_layers=1)
        self.bn2 = nn.BatchNorm1d(dim)

        self.nn3 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv3 = GIN(in_channels=dim, hidden_channels=dim, out_channels=dim, num_layers=1)
        self.bn3 = nn.BatchNorm1d(dim)

        self.nn4 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv4 = GIN(in_channels=dim, hidden_channels=dim, out_channels=dim, num_layers=1)
        self.bn4 = nn.BatchNorm1d(dim)

        self.nn5 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv5 = GIN(in_channels=dim, hidden_channels=dim, out_channels=dim, num_layers=1)
        self.bn5 = nn.BatchNorm1d(dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.nn1(x)
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_attr))
        x = self.bn1(x)
        x = self.nn2(x)
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_attr))
        x = self.bn2(x)
        x = self.nn3(x)
        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_attr))
        x = self.bn3(x)
        x = self.nn4(x)
        x = F.relu(self.conv4(x, edge_index, edge_weight=edge_attr))
        x = self.bn4(x)
        x = self.nn5(x)
        x = F.relu(self.conv5(x, edge_index, edge_weight=edge_attr))
        x = self.bn5(x)
        x = global_mean_pool(x, batch)
        graph_embeds = F.relu(self.fc1(x))
        logit_embeds = self.fc2(x)
        return logit_embeds, None, graph_embeds

### FGWMixup
class FGWMixup(nn.Module):
    def __init__(self, baseline_config):
        super(FGWMixup, self).__init__()
        num_features = baseline_config['in_dim']
        dim = baseline_config['hid_dim']
        num_classes = baseline_config['out_dim']

        self.nn1 = nn.Sequential(nn.Linear(num_features, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv1 = GIN(in_channels=dim, hidden_channels=dim, out_channels=dim, num_layers=1)
        self.bn1 = nn.BatchNorm1d(dim)

        self.nn2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv2 = GIN(in_channels=dim, hidden_channels=dim, out_channels=dim, num_layers=1)
        self.bn2 = nn.BatchNorm1d(dim)

        self.nn3 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv3 = GIN(in_channels=dim, hidden_channels=dim, out_channels=dim, num_layers=1)
        self.bn3 = nn.BatchNorm1d(dim)

        self.nn4 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv4 = GIN(in_channels=dim, hidden_channels=dim, out_channels=dim, num_layers=1)
        self.bn4 = nn.BatchNorm1d(dim)

        self.nn5 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv5 = GIN(in_channels=dim, hidden_channels=dim, out_channels=dim, num_layers=1)
        self.bn5 = nn.BatchNorm1d(dim)

        self.nn6 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv6 = GIN(in_channels=dim, hidden_channels=dim, out_channels=dim, num_layers=1)
        self.bn6 = nn.BatchNorm1d(dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.nn1(x)
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_attr))
        x = self.bn1(x)
        x = self.nn2(x)
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_attr))
        x = self.bn2(x)
        x = self.nn3(x)
        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_attr))
        x = self.bn3(x)
        x = self.nn4(x)
        x = F.relu(self.conv4(x, edge_index, edge_weight=edge_attr))
        x = self.bn4(x)
        x = self.nn5(x)
        x = F.relu(self.conv5(x, edge_index, edge_weight=edge_attr))
        x = self.bn5(x)
        x = self.nn6(x)
        x = F.relu(self.conv6(x, edge_index, edge_weight=edge_attr))
        x = self.bn6(x)
        output, counts = torch.unique(batch, return_counts=True)
        
        x_list = []
        idx = 0
        for num_node in counts:
            idx += num_node
            x_list.append(x[idx-1, :])
        x = torch.stack(x_list, dim=0)

        x = F.dropout(x, p=0.5, training=self.training)
        graph_embeds = F.relu(self.fc1(x))
        x = F.dropout(graph_embeds, p=0.5, training=self.training)
        logit_embeds = self.fc2(x)
        return logit_embeds, None, graph_embeds

### LRGNN
class LRGNNReadout_func(nn.Module):
    def __init__(self, readout_op, hidden):
        super(LRGNNReadout_func, self).__init__()
        self.readout_op = readout_op
        self.readout = global_add_pool

    def reset_parameters(self):
        pass
    
    def forward(self, x, batch):
        x = self.readout(x, batch)
        return x

LRGNNREADOUT_OPS = {
    "global_sum": lambda hidden: LRGNNReadout_func('add', hidden)
}

class LRGNNReadoutOp(nn.Module):
    def __init__(self, primitive, hidden):
        super(LRGNNReadoutOp, self).__init__()
        self._op = LRGNNREADOUT_OPS[primitive](hidden)
    def reset_parameters(self):
        self._op.reset_parameters()
    def reset_params(self):
        self._op.reset_params()

    def forward(self, x, batch):
        return self._op(x, batch)

class LRGNNLaAggregator(nn.Module):

    def __init__(self, mode, hidden_size, num_layers=3):
        super(LRGNNLaAggregator, self).__init__()
        self.mode = mode
        if mode in ['lstm', 'cat', 'max']:
            self.jump = JumpingKnowledge(mode, hidden_size, num_layers=num_layers)
        elif mode == 'att':
            self.att = nn.Linear(hidden_size, 1)

        if mode == 'cat':
            self.lin = nn.Linear(hidden_size * num_layers, hidden_size)
        else:
            self.lin = nn.Linear(hidden_size, hidden_size)
    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.mode in ['lstm', 'cat', 'max']:
            self.jump.reset_parameters()
        if self.mode == 'att':
            self.att.reset_parameters()

    def forward(self, xs):
        if self.mode in ['lstm', 'cat', 'max']:
            output = self.jump(xs)
        elif self.mode == 'sum':
            output = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            output = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'att':
            input = torch.stack(xs, dim=-1).transpose(1, 2)
            weight = self.att(input)
            weight = F.softmax(weight, dim=1)
            output = torch.mul(input, weight).transpose(1, 2).sum(dim=-1) 

        return self.lin(F.relu(output))

LRGNNFF_OPS = {
    'sum': lambda hidden_size, num_layers: LRGNNLaAggregator('sum', hidden_size, num_layers),
    'mean': lambda hidden_size, num_layers: LRGNNLaAggregator('mean', hidden_size, num_layers),
    'max': lambda hidden_size, num_layers: LRGNNLaAggregator('max', hidden_size, num_layers),
    'concat': lambda hidden_size, num_layers: LRGNNLaAggregator('cat', hidden_size, num_layers),
    'lstm': lambda hidden_size, num_layers: LRGNNLaAggregator('lstm', hidden_size, num_layers),
    'att': lambda hidden_size, num_layers: LRGNNLaAggregator('att', hidden_size, num_layers)
}

class LRGNNLaOp(nn.Module):
    def __init__(self, primitive, hidden_size,num_layers=None):
        super(LRGNNLaOp, self).__init__()
        self._op = LRGNNFF_OPS[primitive](hidden_size, num_layers)
    def reset_parameters(self):
        self._op.reset_parameters()
    def forward(self, x):
        return F.relu(self._op(x))

class LRGNNIdentity(nn.Module):

    def __init__(self):
        super(LRGNNIdentity, self).__init__()

    def forward(self, x):
        return x

class LRGNNZero(nn.Module):

    def __init__(self):
        super(LRGNNZero, self).__init__()

    def forward(self, x):
        return x.mul(0.)

LRGNNSC_OPS={
    'zero': lambda: LRGNNZero(),
    'identity': lambda: LRGNNIdentity(),
}

class LRGNNScOp(nn.Module):
    def __init__(self, primitive):
        super(LRGNNScOp, self).__init__()
        self._op = LRGNNSC_OPS[primitive]()

    def forward(self, x):
        return self._op(x)


class LRGNNNaAggregator(nn.Module):
    def __init__(self, in_dim, out_dim, aggregator):
        super(LRGNNNaAggregator, self).__init__()
        if 'gcn' == aggregator:
            self._op = GCNConv(in_dim, out_dim)

    def reset_parameters(self):
        self._op.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self._op(x, edge_index, edge_weight=edge_attr)


LRGNNNA_OPS = {
    'gcn': lambda in_dim, out_dim: LRGNNNaAggregator(in_dim, out_dim, 'gcn')
}

class LRGNNNaOp(nn.Module):
    def __init__(self, primitive, in_dim, out_dim, with_linear=False):
        super(LRGNNNaOp, self).__init__()

        self._op = LRGNNNA_OPS[primitive](in_dim, out_dim)
        self.op_linear = nn.Linear(in_dim, out_dim)
        self.with_linear = with_linear

    def reset_parameters(self):
        self._op.reset_parameters()
        self.op_linear.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        if self.with_linear:
            return self._op(x, edge_index, edge_attr) + self.op_linear(x)
        else:
            return self._op(x, edge_index, edge_attr)

class LRGNN(nn.Module):

    def __init__(self, baseline_config):
        super(LRGNN, self).__init__()
        self.genotype = baseline_config['arch']
        in_dim = baseline_config['in_dim']
        out_dim = baseline_config['out_dim']
        hidden_size = baseline_config['hid_dim']
        self.dropout = baseline_config['dropout']

        self.num_blocks = baseline_config['num_blocks']
        self.num_cells = baseline_config['num_cells']
        self.cell_mode = baseline_config['cell_mode']
        self.BN = baseline_config['BN']
        self.LN = baseline_config['LN']

        ops = self.genotype.split('||')

        self.lin1 = nn.Linear(in_dim, hidden_size)

        self.gnn_layers = nn.ModuleList(
            [LRGNNNaOp(ops[i], hidden_size, hidden_size) for i in range(self.num_blocks)])

        num_node_per_cell = int(self.num_blocks / self.num_cells)
        self.num_node_per_cell = num_node_per_cell

        if self.cell_mode == 'full':
            num_searched_skip = (self.num_blocks + 2) * (self.num_blocks + 1) / 2
        else:
            num_searched_skip = self.num_cells * (num_node_per_cell + 2) * (num_node_per_cell + 1) / 2

        self.num_edges = int(num_searched_skip)
        self.skip_op = nn.ModuleList()
        for i in range(self.num_edges):
            self.skip_op.append(LRGNNScOp(ops[self.num_blocks + i]))

        self.fuse_funcs = nn.ModuleList()
        start = self.num_edges + self.num_blocks
        for i in range(self.num_blocks + self.num_cells):
            if self.cell_mode == 'full':
                input_blocks = i + 1
            else:
                input_blocks = i % (num_node_per_cell + 1) + 1
            self.fuse_funcs.append(LRGNNLaOp(ops[start + i], hidden_size, num_layers=input_blocks))

        self.cell_output_lins = nn.ModuleList()
        for i in range(self.num_cells):
            self.cell_output_lins.append(nn.Linear(hidden_size, hidden_size))

        self.readout_layers = LRGNNReadoutOp(ops[-1], hidden_size)
        self.readout_lin = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, out_dim)

        self.lns = nn.ModuleList()
        if self.LN:
            for i in range(self.num_blocks):
                self.lns.append(LayerNorm(hidden_size))

        self.bns = nn.ModuleList()
        if self.BN:
            for i in range(self.num_blocks):
                self.bns.append(BatchNorm(hidden_size))

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for agg in self.gnn_layers:
            agg.reset_parameters()
        for ff in self.fuse_funcs:
            ff.reset_parameters()
        for lin in self.cell_output_lins:
            lin.reset_parameters()
        self.readout_layers.reset_parameters()
        self.readout_lin.reset_parameters()
        self.classifier.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

        
    def _get_edge_id(self, cell, cur_node, input_node):
        edge_id = (cur_node + 1) * cur_node / 2 + input_node
        return int(edge_id)

    def _get_ff_id(self, cell, cur_node):
        return cell * (self.num_node_per_cell + 1) + cur_node

    def forward(self, x, edge_index, edge_attr, batch):
        cell_output = []
        
        features = []

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        features += [x]
        cell_output += [x]
        num_node_per_cell = int(self.num_blocks / self.num_cells)
        for cell in range(self.num_cells):
            for node in range(num_node_per_cell + 1):
                layer_input = []
                for i in range(node + 1):
                    edge_id = self._get_edge_id(cell, node, i)
                    layer_input += [self.skip_op[edge_id](features[i])]

                ff_id = self._get_ff_id(cell, node)
                tmp_input = self.fuse_funcs[ff_id](layer_input)

                agg_id = cell * self.num_node_per_cell + node
                if node == self.num_node_per_cell:
                    x = self.cell_output_lins[cell](tmp_input)
                else:
                    x = self.gnn_layers[agg_id](tmp_input, edge_index, edge_attr)
                x = F.relu(x)

                if node != self.num_node_per_cell:
                    if self.BN:
                        x = self.bns[agg_id](x)
                    elif self.LN:
                        x = self.lns[agg_id](x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                features += [x]

            features = [x]
            cell_output += [x]

        output = self.readout_layers(x, batch)
        output = F.relu(self.readout_lin(output))
        graph_embeds = F.dropout(output, p=self.dropout, training=self.training)
        logit_embeds = self.classifier(graph_embeds)
        return logit_embeds, None, graph_embeds

### GmapAD
class GmapAD(nn.Module):
    def __init__(self, baseline_config):
        super(GmapAD, self).__init__()
        in_dim = baseline_config['in_dim']
        hid_dim = baseline_config['hid_dim']
        out_dim = baseline_config['out_dim']
        dropout = baseline_config['dropout']
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, int(hid_dim/2))
        self.fc = nn.Linear(int(hid_dim/2), out_dim)
        self.pool = global_mean_pool
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batchind):
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        n_reps = self.conv2(x, edge_index, edge_weight=edge_attr)
        node_embeds = n_reps.relu()

        g_reps = self.pool(node_embeds, batchind)
        graph_embeds = F.dropout(g_reps, p=self.dropout)
        logit_embeds = self.fc(graph_embeds)
        return logit_embeds, node_embeds, graph_embeds

### GRDL
class GRDLMLP(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers, device):
        super().__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.linears.append(nn.Linear(input_channels, hidden_channels).to(device))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for layer in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_channels, hidden_channels).to(device))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.linears.append(nn.Linear(hidden_channels, output_channels).to(device))

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.linears[i](x)
            x = self.bns[i](x)
            x = F.leaky_relu(x, negative_slope=0.1)
        x = self.linears[-1](x)
        return x
    
    def reset_parameters(self):
        for layer in self.linears:
            layer.reset_parameters()
        for layer in self.bns:
            layer.reset_parameters()

class GRDLGINExtractor(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 num_layer_mlp,
                 num_layer_gin,
                 jump_mode, device):
        super().__init__()
        if jump_mode is not None:
            self.jump_layer = JumpingKnowledge(jump_mode)
        else:
            self.jump_layer = None
        self.GIN_layers = nn.ModuleList()
        self.MLP_layers = nn.ModuleList()
        self._num_layer_gin = num_layer_gin
        for layer in range(num_layer_gin):
            if layer == 0:
                local_input_channels = input_channels
            else:
                local_input_channels = hidden_channels
            self.MLP_layers.append(GRDLMLP(local_input_channels, hidden_channels, hidden_channels, num_layer_mlp, device))
            self.GIN_layers.append(GIN(in_channels=hidden_channels, hidden_channels=hidden_channels, out_channels=hidden_channels, num_layers=1))
    def forward(self, x, edge_index, edge_attr):
        xs = []
        curr_x = x
        for i in range(self._num_layer_gin):
            curr_x = self.MLP_layers[i](curr_x)
            curr_x = self.GIN_layers[i](x=curr_x, edge_index=edge_index, edge_weight=edge_attr)
            xs.append(curr_x)
        if self.jump_layer is None:
            return xs[-1]
        else:
            return self.jump_layer(xs)
    
    def reset_parameters(self):
        for layer in self.GIN_layers:
            layer.reset_parameters()

class GRDLReferenceLayer(torch.nn.Module):
    def __init__(self, input_channels, output_channels, num_supp, gamma, device):
        super().__init__()
        self.output_channels = output_channels
        self.init_gamma = gamma
        self.init_atoms = None
        self.atoms = Parameter(torch.empty(size=(output_channels, num_supp, input_channels)))
        self.gamma = Parameter(torch.empty(size=(1,)))
        self.reset_parameters()
        self.device = device

    def init_atoms(self, num_supp, random_init, init_atoms):
        if random_init:
            atoms = Parameter(torch.randn(size=(self.output_channels, num_supp, self.input_channels)))
        else:
            if isinstance(init_atoms, list):
                atoms = [Parameter(atom) for atom in init_atoms]
            else:
                atoms = Parameter(init_atoms)
        return atoms
    
    def cal_mmd(self, atoms, x, batch, gamma):
        d_xy = ((atoms[:, :, np.newaxis, :] - x[np.newaxis, np.newaxis, :, :])**2).sum(axis=-1)
        k_xy = self.neg_exp(d_xy, gamma)
        d_yy = ((atoms[:, :, np.newaxis, :] - atoms[:, np.newaxis, :, :])**2).sum(axis=-1)
        k_yy = self.neg_exp(d_yy, gamma)
        d_xx = ((x[:, np.newaxis, :] - x)**2).sum(axis=-1)
        k_xx = self.neg_exp(d_xx, gamma)
    
        unique_batch, num_element = torch.unique(batch,return_counts=True)
        num_element = num_element.cpu()
        index = torch.hstack((torch.tensor(0), torch.cumsum(num_element, 0)))
        A = torch.zeros(batch.shape[0], unique_batch.shape[0])
        for i in range(unique_batch.shape[0]):
            A[index[i]: index[i+1], i] = torch.ones(num_element[i])/ num_element[i]
        A = A.to(x.device)
        xx = torch.diag(A.T @ k_xx @ A)
        xy = torch.mean(k_xy @ A, dim=1)
        yy = k_yy.mean(dim=1).mean(dim=1)
        mmd_distance = (yy.reshape(-1, 1) + xx - 2 * xy).T 
        mmd_distance = mmd_distance ** 0.5
        return mmd_distance

    def neg_exp(self, dist, gamma):
        return torch.exp(-dist/gamma)

    def forward(self, x, batch):
        x.to(self.device)
        batch.to(self.device)
        mmd_distance = self.cal_mmd(self.atoms, x, batch, self.gamma)
        S = -mmd_distance
        y_hat = torch.exp(S)
        result = y_hat / y_hat.sum(dim=1, keepdim=True)
        return result
    
    def discriminate_loss(self):
        num_atoms = self.output_channels
        loss = 0
        for i in range(num_atoms-1):
            x = self.atoms[i]
            y = self.atoms[i+1:]
            d_xy = ((x[np.newaxis, :, np.newaxis, :] - y[:, np.newaxis, :, :])**2).sum(dim=-1)
            k_xy = self.neg_exp(d_xy, self.gamma)
            d_xx = ((x[:, np.newaxis, :] - x[np.newaxis, :, :])**2).sum(dim=-1)
            k_xx = self.neg_exp(d_xx, self.gamma)
            d_yy = ((y[:, :, np.newaxis, :] - y[:, np.newaxis, :, :])**2).sum(dim=-1)
            k_yy = self.neg_exp(d_yy, self.gamma)
            mmd_distance = k_xx.mean() + k_yy.mean(dim=-1).mean(dim=-1) - 2*k_xy.mean(dim=-1).mean(dim=-1)
            loss += -(mmd_distance**0.5).sum()        
        return loss
    
    def reset_parameters(self):
        
        if self.init_atoms is not None:
            self.atoms = self.init_atoms
        else:
            self.atoms.data.normal_(0, 1)
        
        self.gamma.data.fill_(self.init_gamma)

class GRDL(nn.Module):
    def __init__(self, baseline_config, info, device):
        super().__init__()
        self.num_atoms = baseline_config['out_dim']
        self.num_atom_supp = info['num_atom_supp']
        self.extractor = GRDLGINExtractor(baseline_config['in_dim'], baseline_config['hid_dim'], baseline_config['ex_num_layer_mlp'], baseline_config['ex_num_layer_gin'], baseline_config['ex_jump_mode'], device)
        self.mmd = GRDLReferenceLayer(baseline_config['hid_dim'], baseline_config['out_dim'], info['num_atom_supp'], baseline_config['gamma'], device)
        self.pool = global_mean_pool
        self.device = device

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.extractor(x, edge_index, edge_attr)
        graph_embeds = self.pool(x, batch)
        logit_embeds = self.mmd(x, batch)
        return logit_embeds, None, graph_embeds

    def reset_parameters(self):
        self.extractor.reset_parameters()
        self.mmd.reset_parameters()

### RQGNN
class RQGNN(nn.Module):
    def __init__(self, baseline_config, device):
        super(RQGNN, self).__init__()
        in_dim = baseline_config['in_dim']
        hid_dim = baseline_config['hid_dim']
        out_dim = baseline_config['out_dim']
        width = baseline_config['width']
        depth = baseline_config['depth']
        dropout = baseline_config['dropout']
        normalize = baseline_config['normalize']

        self.conv = []
        for i in range(width):
            self.conv.append(ChebConv(in_dim, in_dim, depth).to(device))

        self.linear = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, in_dim)
        self.linear3 = nn.Linear(in_dim * len(self.conv), hid_dim)
        self.linear4 = nn.Linear(hid_dim, hid_dim)
        self.act = nn.LeakyReLU()

        self.linear5 = nn.Linear(in_dim, hid_dim)
        self.linear6 = nn.Linear(hid_dim, hid_dim)
        
        self.linear7 = nn.Linear(hid_dim * 2, out_dim)

        self.bn = nn.BatchNorm1d(hid_dim * 2)

        self.dp = nn.Dropout(p=dropout)
        self.normalize = normalize

        self.linear8 = nn.Linear(in_dim, hid_dim)
        self.linear9 = nn.Linear(hid_dim, hid_dim)
        self.pool = global_add_pool

    def forward(self, x, edge_index, edge_attr, batchind, xlxs, node_belongs):
        h = self.linear(x)
        h = self.act(h)

        h = self.linear2(h)
        h = self.act(h)

        h_final = []
        for conv in self.conv:
            h0 = conv(h, edge_index, edge_weight=edge_attr)
            h_final.append(h0)
        
        h_final = torch.cat(h_final, dim=-1)

        h = self.linear3(h_final)
        h = self.act(h)
        
        h = self.linear4(h)
        h = self.act(h)

        tmpscores = self.linear8(xlxs)
        tmpscores = self.act(tmpscores)
        tmpscores = self.linear9(tmpscores)
        tmpscores = self.act(tmpscores)
        scores = []
        for i, node_belong in enumerate(node_belongs):
            scores.append(torch.unsqueeze(torch.mv(h[node_belong], tmpscores[i]), 1))
        scores = torch.cat(scores, dim=0)

        h = h * scores

        h = self.pool(h, batchind)
        
        xLx = self.linear5(xlxs)
        
        xLx = self.linear6(xLx)
        xLx = self.act(xLx)

        h = torch.cat([h, xLx], -1)

        if self.normalize:
            h = self.bn(h)

        graph_embeds = self.dp(h)
        logit_embeds = self.linear7(graph_embeds)

        return logit_embeds, None, graph_embeds

#UniGAD
class UniGADGCN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 encoding=False
                 ):
        super(UniGADGCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = 'leakyrelu'
        self.dropout = dropout

        last_activation = self.activation if encoding else None
        
        self.gcn_layers.append(GCN(in_channels=in_dim, hidden_channels=num_hidden, out_channels=num_hidden, num_layers=1, act=self.activation))
        for l in range(1, num_layers - 1):
            self.gcn_layers.append(GCN(in_channels=num_hidden, hidden_channels=num_hidden, out_channels=num_hidden, num_layers=1, act=self.activation))
        self.gcn_layers.append(GCN(in_channels=num_hidden, hidden_channels=num_hidden, out_channels=out_dim, num_layers=1, act=last_activation))

        self.head = nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        h = x
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](x=h, edge_index=edge_index, edge_weight=edge_attr)
            hidden_list.append(h)
        
        return self.head(h)

class UniGADGraphMAE(nn.Module):
    def __init__(self, baseline_config, device):
        super(UniGADGraphMAE, self).__init__()
        self._mask_ratio = baseline_config['mask_ratio']

        self._output_hidden_size = baseline_config['hid_dim']

        self._replace_ratio = baseline_config['replace_ratio']
        self._mask_token_rate = 1 - self._replace_ratio

        in_dim = baseline_config['in_dim']
        hid_dim = baseline_config['hid_dim']
        num_layer = baseline_config['num_layer_pretrain']
        drop_ratio = baseline_config['dropout']
        alpha_l = baseline_config['alpha_l']

        enc_in_dim = in_dim
        enc_hid_dim = hid_dim
        enc_out_dim = hid_dim

        dec_in_dim = hid_dim
        dec_hid_dim = hid_dim
        dec_out_dim = in_dim

        self.in_dim = in_dim
        self.embed_dim = enc_out_dim

        self.encoder = UniGADGCN(
                in_dim=enc_in_dim,
                num_hidden=enc_hid_dim,
                out_dim=enc_out_dim,
                num_layers=num_layer,
                dropout=drop_ratio,
                encoding = True,
            )
        
        self.decoder = UniGADGCN(
                in_dim=dec_in_dim,
                num_hidden=dec_hid_dim,
                out_dim=dec_out_dim,
                num_layers=num_layer,
                dropout=drop_ratio,
                encoding = False,
            )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.encoder_to_decoder = nn.Linear(enc_out_dim, dec_in_dim, bias=False)

        self.criterion = partial(self.sce_loss, alpha=alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def sce_loss(self, x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss

    def encoding_mask_noise(self, x, mask_ratio=0.3):
        num_nodes = len(x)
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_ratio * num_nodes)

        num_mask_nodes = int(mask_ratio * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, mask_nodes

    def forward(self, x, edge_index, edge_attr):
        loss = self.mask_attr_prediction(x, edge_index, edge_attr)
        return loss

    def mask_attr_prediction(self, x, edge_index, edge_attr):
        use_x, mask_nodes = self.encoding_mask_noise(x, self._mask_ratio)

        enc_rep = self.encoder(x, edge_index, edge_attr)
        rep = self.encoder_to_decoder(enc_rep)

        recon = self.decoder(rep, edge_index, edge_attr)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss

    def embed(self, x, edge_index, edge_attr):
        return self.encoder(x, edge_index, edge_attr)

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])


class UniGADMLP_E2E(nn.Module):
    def __init__(self, pretrain_model, baseline_config, device):
        super(UniGADMLP_E2E, self).__init__()
        num_classes = baseline_config['out_dim']
        embed_dims = baseline_config['hid_dim']
        dropout_rate = baseline_config['dropout']
        self.dropout = nn.Dropout(dropout_rate, inplace=True) if dropout_rate > 0 else nn.Identity()
        stitch_mlp_layers = baseline_config['stitch_mlp_layers']
        final_mlp_layers = baseline_config['final_mlp_layers']
        self.pretrain_model = pretrain_model

        self.act = nn.ReLU()
        self.layer1 = nn.Sequential()
        for _ in range(stitch_mlp_layers):
            self.layer1.append(nn.Linear(embed_dims, embed_dims))
            self.layer1.append(self.act)
        self.layer2 = nn.Parameter(data=torch.ones(1), requires_grad=True)
        self.layer3 = nn.Sequential()
        for _ in range(stitch_mlp_layers):
            self.layer3.append(nn.Linear(embed_dims, embed_dims))
            self.layer3.append(self.act)
        self.layer4 = nn.Parameter(data=torch.ones(1), requires_grad=True)
        self.layer56 = nn.Sequential(self.dropout) 
        for _ in range(final_mlp_layers):
            self.layer56.append(nn.Linear(embed_dims, embed_dims))
            self.layer56.append(self.act)
        self.layer56.append(nn.Linear(embed_dims, num_classes))

        self.pool = global_mean_pool

    def forward(self, x, edge_index, edge_attr, batchind):
        h = self.pretrain_model.embed(x, edge_index, edge_attr)
        graph_embeds = self.pool(h, batchind)

        logit_embeds = self.layer2 * self.layer1(graph_embeds)
        logit_embeds = self.layer4 * self.layer3(logit_embeds)
        logit_embeds = self.layer56(logit_embeds)
        return logit_embeds, None, graph_embeds
