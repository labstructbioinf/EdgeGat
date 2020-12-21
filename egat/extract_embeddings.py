import torch as th
import dgl
import numpy as np

from edge_gat_learnable import MultiHeadEGATLayer as GATConv
class ImportanceTracker:
    @classmethod
    def get_forward_outputs(cls, edge_gat_layer, sample_tuple):
        #regular forward pass
        nodes, edges = edge_gat_layer(**sample_tuple)
        edges = edge_gat_layet.attn_fc_coef(edges)
        
        return nodes, edges