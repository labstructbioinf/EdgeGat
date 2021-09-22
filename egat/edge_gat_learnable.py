import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadEGATLayer(nn.Module):
    """
    Parameters
    ----------
    in_node_feats : int, or pair of ints
        Input node feature size.
    in_edge_feats : int, or pair of ints
        Input edge feature size.
    out_node_feats : int
        Output nodes feature size.
    out_edge_feats : int
        Output edge feature size.
    num_heads : int
        Number of attention heads.

    """
    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 num_heads,
                 attention_scaler = 'softmax',
                 **kw_args):
        
        super().__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_nodes = nn.Linear(in_node_feats, out_node_feats*num_heads, bias=True)
        self.fc_edges = nn.Linear(in_edge_feats + 2*in_node_feats, out_edge_feats*num_heads, bias=True)
        self.fc_attn = nn.Linear(out_edge_feats, num_heads, bias=False)
        self.reset_parameters()
        self.attention_scaler = attention_scaler
        if self.attention_scaler == 'sigmoid':
            self.scaler = nn.Sigmoid()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_nodes.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edges.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_attn.weight, gain=gain)

    def edge_attention(self, edges):
        #extract features
        h_src = edges.src['h']
        h_dst = edges.dst['h']
        f = edges.data['f']
        stack = torch.cat([h_src, f, h_dst], dim=-1)
        # apply FC and activation
        f_out = self.fc_edges(stack)
        f_out = F.leaky_relu(f_out)
        f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
        # FC to reduce edge_feats to scalar
        a = self.fc_attn(f_out).sum(-1).unsqueeze(-1)
        #print(a.shape)
        return {'a': a, 'f' : f_out}

    def message_func(self, edges):
        return {'h': edges.src['h'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        if self.attention_scaler == 'sigmoid':
            alpha = self.scaler(nodes.mailbox['a'])
        else:
            alpha = F.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'h': h}

    def forward(self, g, nfeats, efeats):
        r"""
        Compute new node and edge features.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        nfeats : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input node feature of shape :math:`(N, D_{in})`
            where:
                :math:`D_{in}` is size of input node feature,
                :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        efeats : torch.Tensor
            
        Returns
        -------
        pair of torch.Tensor
            node output features followed by edge output features
            The node output feature of shape :math:`(N, H, D_{out})` 
            The edge output feature of shape :math:`(S, H, F_{out})`
            where:
                :math:`H` is the number of heads,
                :math:`D_{out}` is size of output node feature,
                :math:`F_{out}` is size of output edge feature.
        """
        ##TODO allow node src and dst feats
        g.edata['f'] = efeats
        g.ndata['h'] = nfeats
        
        g.apply_edges(self.edge_attention)
        
        nfeats_ = self.fc_nodes(nfeats)
        nfeats_ = nfeats_.view(-1, self._num_heads, self._out_node_feats)
        
        g.ndata['h'] = nfeats_
        g.update_all(message_func = self.message_func,
                     reduce_func = self.reduce_func)
        
        return g.ndata.pop('h'), g.edata.pop('f')
    