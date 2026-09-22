from frameworks import *

framework_dict = {
    'GCN': GNN_framework,
    'SAGE': GNN_framework,
    'GAT': GNN_framework,
    'GIN': GNN_framework,
    'iGAD': iGAD_framework, 
    'NS': NS_framework,
    'GLA': GLA_framework,
    'GMixup': GMixup_framework,
    'FGWMixup': FGWMixup_framework, 
    'LRGNN': LRGNN_framework,
    'GmapAD': GmapAD_framework,
    'GRDL': GRDL_framework, 
    'RQGNN': RQGNN_framework,
    'UniGAD': UniGAD_framework
}

baseline_configs = {}

baseline_configs['GCN'] = {
    'hid_dim': 64, 
    'layer_num': 2, 
    'lr': 1e-2, 
    'nepoch': 100, 
    'batchsize': 512, 
    'dropout': 0.1, 
    'decay': 1e-5
}

baseline_configs['SAGE'] = {
    'hid_dim': 128,
    'layer_num': 2,
    'lr': 1e-2,
    'nepoch': 100,
    'batchsize': 512,
    'dropout': 0, 
    'decay': 1e-5
}

baseline_configs['GAT'] = {
    'hid_dim': 256,
    'layer_num': 3,
    'lr': 1e-2,
    'nepoch': 100,
    'batchsize': 512,
    'dropout': 0,
    'heads': 4, 
    'decay': 0
}

baseline_configs['GIN'] = {
    'hid_dim': 64,
    'layer_num': 5,
    'lr': 1e-2,
    'nepoch': 100,
    'batchsize': 32,
    'dropout': 0, 
    'eps': 0,
    'decay': 0
}

baseline_configs['iGAD'] = {
    'dim_1': 32, 
    'f_hidden_dim': 64, 
    'f_output_dim': 32, 
    't_hidden_dim': 32, 
    't_output_dim': 16, 
    'lr': 1e-3,
    'nepoch': 100, 
    'batchsize': 128,
    'dropout': 0.2, 
    'normalize': 1,
    'n_subgraphs': 5, 
    'size_subgraphs': 10, 
    'max_step': 6
}


baseline_configs['NS'] = {
    'hid_dim': 64,
    'layer_num': 5,
    'lr': 1e-2,
    'nepoch': 100,
    'batchsize': 128,
    'dropout': 0.5,
    'mlp_layers': 2
}

baseline_configs['GLA'] = {
    'hid_dim': 64,
    'lr': 1e-3,
    'nepoch': 100,
    'batchsize': 256,
    'dropout': 0,
    'num_feat_layers': 1,
    'residual': 0,
    'num_conv_layers': 3,
    'num_fc_layers': 2,
    'edge_norm': 1, 
    'eta': 1
}

baseline_configs['GMixup'] = {
    'hid_dim': 64,
    'lr': 1e-2,
    'nepoch': 100, 
    'batchsize': 32,
    'decay': 5e-4
}

baseline_configs['FGWMixup'] = {
    'hid_dim': 64,
    'lr': 1e-2,
    'nepoch': 100,
    'batchsize': 32,
    'decay': 5e-4
}


baseline_configs['LRGNN'] = {
    'hid_dim': 256,
    'lr': 0.0038948647662910996, 
    'nepoch': 100, 
    'batchsize': 128,
    'dropout': 0.2,
    'decay': 0.0006578269294157753,
    'arch': 'gcn||gcn||gcn||gcn||gcn||gcn||gcn||gcn||identity||zero||identity||zero||zero||zero||zero||zero||identity||zero||zero||zero||zero||identity||identity||identity||identity||zero||identity||identity||identity||identity||zero||identity||zero||identity||zero||identity||zero||zero||identity||zero||identity||zero||zero||zero||identity||identity||identity||identity||identity||zero||zero||identity||zero||mean||max||lstm||lstm||max||mean||concat||lstm||att||global_sum', 
    'num_blocks': 8, 
    'num_cells': 1, 
    'cell_mode': 'full', 
    'BN': True, 
    'LN': False
}


baseline_configs['GmapAD'] = {
    'hid_dim': 128,
    'lr': 1e-3,
    'nepoch': 100, 
    'batchsize': 512,
    'dropout': 0.5, 
    'k': 64, 
    'cand_size': 30, 
    'evo_nepoch': 200,
    'mut_rate': 0.5,
    'cros_rate': 0.9
}

baseline_configs['GRDL'] = {
    'hid_dim': 64,
    'lr1': 1e-3,
    'lr2': 1e-1,
    'nepoch': 100,
    'batchsize': 32,
    'ex_num_layer_mlp': 2,
    'ex_num_layer_gin': 5,
    'ex_jump_mode': 'max',
    'num_atom_supp': 'G3',
    'gamma': 500, 
    'lam': 1
}

baseline_configs['RQGNN'] = {
    'hid_dim': 64, 
    'width': 4, 
    'depth': 6,
    'lr': 5e-3,
    'nepoch': 100, 
    'batchsize': 512, 
    'dropout': 0.4, 
    'normalize': 1,
    'beta': 0.999,
    'gamma': 1.5
}

baseline_configs['UniGAD'] = {
    'hid_dim': 32,
    'lr': 3e-3,
    'nepoch': 100, 
    'batchsize': 128,
    'dropout': 0,
    'mask_ratio': 0.5,
    'replace_ratio': 0,
    'num_layer_pretrain': 2,
    'alpha_l': 2,
    'stitch_mlp_layers': 1,
    'final_mlp_layers': 2,
    'pretrain': 'GraphMAE'
}
