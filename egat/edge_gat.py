import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim_n, in_dim_e, out_dim, use_sigmoid = True):
        super().__init__()
        self.fc = nn.Linear(in_dim_n, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim + in_dim_e, 1, bias=False)
        self.use_sigmoid = use_sigmoid
        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        src_data = edges.src['z']
        dst_data = edges.dst['z']
        feat_data = edges.data['feats']
        stacked = torch.cat([src_data, feat_data, dst_data], dim=1)
        #print(src_data.size(), dst_data.size(), feat_data.size())
        a = self.attn_fc(stacked)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        #print(nodes.mailbox['e'].size())
        #alpha = 2*self.sigmoid(nodes.mailbox['e']) - 1
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, nfeats):
        z = self.fc(nfeats)
        g.ndata['z'] = z
        #g.edata['feats'] = efeats
        g.apply_edges(self.edge_attention)
        g.update_all(message_func = self.message_func,
                     reduce_func = self.reduce_func)
        return g.ndata.pop('h')
    
    
class MultiHeadEGATLayer(nn.Module):
    def __init__(self, in_dim_n, in_dim_e, out_dim, num_heads, activation=None, **kw_args):
        super().__init__()
        self.heads = nn.ModuleList()
        self.out_dim = out_dim
        self.activation = activation
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim_n, in_dim_e, out_dim))
        
    def forward(self, g, nfeats):
        head_outs = [attn_head(g, nfeats) for attn_head in self.heads]
        head_outs = torch.cat(head_outs, dim=1)
        if self.activation is not None:
            return F.leaky_relu(head_outs)
        return head_outs