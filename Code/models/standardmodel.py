from torch.nn import Module, Sequential, ReLU, Dropout, BatchNorm1d
import torch.nn.functional as F
import torch.nn as nn 
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import  Linear
import torch 
from utilities.tam import * 
from types import SimpleNamespace

class NodeGNN(Module):
    def __init__(self, layer, edge_type, hidden_channels, dropout, output_channels = 2):
        super().__init__()
        self.conv1 = layer(-1, hidden_channels) # lazy load
        self.conv2 = layer(hidden_channels, hidden_channels//2)
        self.fc = nn.Linear(hidden_channels//2, output_channels)
        self.dropout = dropout
        self.edge_type = edge_type
        if edge_type == 'weight':
            self.weight_lr = Linear(-1, 1)

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        if self.edge_type =='nan':
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc(x)
        elif self.edge_type =='weight':
            assert edge_attr is not None
            w = F.relu(self.weight_lr(edge_attr))
            x = self.conv1(x, edge_index, w)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, w)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc(x)
        elif self.edge_type =='attr':
            assert edge_attr is not None
            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc(x)
        return x
    

class GraphGNN(Module):
    def __init__(self, layer, 
                 edge_type,
                 internal_hidden_channels, 
                 internal_dropout,
                 output_hidden_channels, 
                 output_dropout, output_channels):
        super().__init__()
        self.model = NodeGNN( layer
                                , edge_type
                                , internal_hidden_channels
                                , internal_dropout
                                , output_hidden_channels
                                )
        # output layer
        self.output_convs =  Sequential(
                                Linear(-1, output_hidden_channels),
                                BatchNorm1d(output_hidden_channels),
                                ReLU(),
                                Dropout(p=output_dropout), 
                                Linear(output_hidden_channels, output_channels)
                            )
        self.edge_type = edge_type

    def forward(self, data, **kwargs):
        x = self.model(data.x, data.edge_index, data.edge_attr)
        batch_x  = torch.cat([gmp(x, data.batch), gap(x, data.batch), data.centroid_feats], dim=1)
        output = self.output_convs(batch_x)
        return output
    



class TAM(nn.Module):
    def __init__(self, layer, edge_type, hidden_channels, dropout, output_channels, temp_phi=1.2, 
                 temp_gamma=0.4, tam_alpha=2.5,
                 tam_beta=0.5, act = 'softplus', include_mask =True, warmup=15):
        super(TAM, self).__init__()
        if edge_type in ['weight', 'attr']:
            self.agg = MultiHeadEdgeAggregation(edge_dim=3,
                                hidden_dim=4,      # tiny MLP
                                act=act)
        else:
            self.agg = MeanAggregation()
       # original paper settings 
        self.default_args = SimpleNamespace(
                tam=True,        # turn TAM on
                temp_phi=temp_phi,    # temperature for φ
                temp_gamma = temp_gamma, # temperature for gamma
                tam_alpha=tam_alpha,   
                tam_beta=tam_beta,   

            )
        self.base_model = NodeGNN( layer, edge_type, hidden_channels, dropout, output_channels)
        self.include_mask = include_mask
        self.warmup = warmup
    
    def reset_stage(self, epoch):
        self.warmup = epoch<=self.warmup
        
    def forward(self, data): # tuning epoches 
        if self.include_mask:
            assert data.mask is not None
            mask = data.mask 
        else: # use all nodes in graph 
            mask = torch.ones(data.x.shape[0]).bool()
        logits = self.base_model(data.x, data.edge_index, data.edge_attr) # original logits
        class_list = [len(data.y == i) for i in list(set(data.y.tolist()))] 
        logits = adjust_output(
                        self.default_args,
                        logits,
                        data.edge_index,
                        data.y, 
                        mask, 
                        self.agg,
                        class_list,
                        self.warmup, 
                        edge_attr = data.edge_attr)
        return logits