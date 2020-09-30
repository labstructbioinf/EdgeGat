import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim_n, in_dim_e, out_dim_n, out_dim_e, use_sigmoid = False, **kw_args):
        
        super().__init__()
        self.fc = nn.Linear(in_dim_n, out_dim_n, bias=False)
        self.attn_fc_edge = nn.Linear(in_dim_e + 2 * out_dim_n, out_dim_e, bias=True)
        self.attn_fc_coef = nn.Linear(out_dim_e, 1, bias=False)
        self.use_sigmoid = use_sigmoid
        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc_coef.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        src_data = edges.src['z']
        dst_data = edges.dst['z']
        feat_data = edges.data['feats']
        #print(src_data.size(), dst_data.size(), feat_data.size())
        stacked = torch.cat([src_data, feat_data, dst_data], dim=1)
        feat_data = self.attn_fc_edge(stacked)
        feat_data = F.leaky_relu(feat_data)
        a = self.attn_fc_coef(feat_data)
        a = F.leaky_relu(a)
        return {'attn': F.leaky_relu(a), 'feats' : feat_data}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'attn': edges.data['attn']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['attn'], dim=1)
        #print(nodes.mailbox['e'].size())
        #alpha = 2*self.sigmoid(nodes.mailbox['e']) - 1
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, nfeats, efeats):
        z = self.fc(nfeats)
        g.edata['feats'] = efeats
        g.ndata['z'] = z
        #g.edata['feats'] = efeats
        g.apply_edges(self.edge_attention)
        g.update_all(message_func = self.message_func,
                     reduce_func = self.reduce_func)
        return g.ndata.pop('h'), g.edata.pop('feats')
    
    
class MultiHeadEGATLayer(nn.Module):
    '''
    Multihead version of Edge-GAT layer
    Tips: 
        * avoid high dimensionality of `out_dim_e` - increases computations time
        * edge features are returned in same form as nodes that is (Batch, num_heads, out_dim_e)
    params:
        are same as in regular dgl.nn.pytorch.GATConv except:
            in_dim_e (int) number of input edge features
            out_dim_e (int) number of output edge features
    returns:
        node_features (torch.Tensor)
        edge_features (torch.Tensor)
    '''
    def __init__(self,  in_dim_n, in_dim_e, out_dim_n, out_dim_e, num_heads, activation=None, **kw_args):
        super().__init__()
        self.heads = nn.ModuleList()
        self.activation = activation
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim_n, in_dim_e, out_dim_n, out_dim_e))
        
    def forward(self, g, nfeats, efeats):
        nodes_stack, edges_stack = [], []
        for attn_head in self.heads:
            nodes, edges = attn_head(g, nfeats, efeats)
            nodes_stack.append(nodes)
            edges_stack.append(edges)
            
        nodes_stack = torch.cat(nodes_stack, dim=1)
        edges_stack = torch.cat(edges_stack, dim=1)
        if self.activation is not None:
            nodes_stack =  F.leaky_relu(nodes_stack)
            edges_stack = F.leaky_relu(edges_stack)
        return nodes_stack, edges_stack