import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F




class GATLayer(nn.Module):
    '''
    single head variation of edge-gat
    '''
    def __init__(self, in_dim_n, in_dim_e, out_dim_n, out_dim_e, attention_scaler = 'softmax', **kw_args):
        
        super().__init__()
        self.fc = nn.Linear(in_dim_n, out_dim_n, bias=False)
        self.attn_fc_edge = nn.Linear(in_dim_e + 2 * out_dim_n, out_dim_e, bias=True)
        self.attn_fc_coef = nn.Linear(out_dim_e, 1, bias=False)
        self.reset_parameters()
        self.attention_scaler = attention_scaler
        if self.attention_scaler == 'sigmoid':
            self.scaler = nn.Sigmoid()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc_coef.weight, gain=gain)

    def edge_attention(self, edges):
        #extract features
        src_data = edges.src['z']
        dst_data = edges.dst['z']
        feat_data = edges.data['feats']
        #merge node_i - edge_ij - node_j features
        stacked = torch.cat([src_data, feat_data, dst_data], dim=1)
        # apply FC and activation
        feat_data = self.attn_fc_edge(stacked)
        feat_data = F.leaky_relu(feat_data)
        # FC to reduce edge_feats to scalar
        a = self.attn_fc_coef(feat_data)
        return {'attn': a, 'feats' : feat_data}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'attn': edges.data['attn']}

    def reduce_func(self, nodes):
        if self.attention_scaler == 'sigmoid':
            alpha = self.scaler(nodes.mailbox['attn'])
        else:
            alpha = F.softmax(nodes.mailbox['attn'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, nfeats, efeats):
        z = self.fc(nfeats)
        g.edata['feats'] = efeats
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(message_func = self.message_func,
                     reduce_func = self.reduce_func)
        return g.ndata.pop('h'), g.edata.pop('feats')
    
    
class MultiHeadEGATLayer(nn.Module):
    '''
    Multihead version of Edge-GAT layer. Variation over Deep Graph Library GAT tutorial:
    https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
    Tips: 
        * avoid high dimensionality of `out_dim_e` - increases computations time
        * edge features are returned in same form as nodes that is (Batch, num_heads, out_dim_e)
    params:
        most of them are same as in regular dgl.nn.pytorch.GATConv
            in_dim_e (int) number of input edge features
            out_dim_e (int) number of output edge features
            in_dim_n (int) number of input node features
            out_dim_n (int) number of output node features
            num_heads (int) number of attention heads 
            activation (None, or torch activation eg: F.relu) default None
                activation after concatenation
            attention_scaler (str) `sigmoid` or `softmax` - tells how to scale attention
                coefficients
            
    returns:
        (node_features, edge_features) (tuple of torch.(cuda)FloatTensor's)
    '''
    def __init__(self,  in_dim_n, in_dim_e, out_dim_n, out_dim_e, num_heads, \
                 activation=None,attention_scaler='softmax', **kw_args):
        super().__init__()
        self.heads = nn.ModuleList()
        self.activation = activation
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim_n, in_dim_e, out_dim_n, out_dim_e, attention_scaler))
        
    def forward(self, g, nfeats, efeats):
        nodes_stack, edges_stack = [], []
        for attn_head in self.heads:
            nodes, edges = attn_head(g, nfeats, efeats)
            nodes_stack.append(nodes.unsqueeze(1))
            edges_stack.append(edges.unsqueeze(1))
            
        nodes_stack = torch.cat(nodes_stack, dim=1)
        edges_stack = torch.cat(edges_stack, dim=1)
        if self.activation is not None:
            nodes_stack =  self.activation(nodes_stack)
            edges_stack = self.activation(edges_stack)
        return nodes_stack, edges_stack
    
            
        