# EdgeGat
Reprository with Graph Attention Layer with fully valuable edge features  \
extension of GAT tutorial: https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html  \
use it as regular torch/dgl layer
## requirements

```
python == 3.7
torch ==  1.7.0
dgl   ==  0.5.3
```

## example use

create graph
```python
import dgl
import networkx as nx #optionally for graph sample creation
import torch as th
from egat import MultiHeadEGATLayer

num_nodes = 45 
num_node_feats = 20
num_edge_feats = 20
num_attn_heads = 1


sample_contacts = th.rand((num_nodes, num_nodes)) #random distance matrix
sample_adj = sample_contacts > 0.5 #binarize to adjecency matrix
sample_nx = nx.Graph(sample_adj.cpu().numpy()) # convert to dgl.Graph object
sample_graph = dgl.from_networkx(sample_nx)    

node_features = th.rand((num_nodes, num_node_feats)) 
edge_features = th.rand((sample_graph.number_of_edges(), num_edge_feats))
```

initialize egat layer

```python
#use as regular torch/dgl layer work similar as GATConv from dgl library
egat = MultiHeadEGATLayer(in_dim_n=num_node_feats,
                          in_dim_e=num_edge_feats,
                          num_heads=num_attn_heads,
                          out_dim_e=10,
                          out_dim_n=10,
                          activation=th.nn.functional.leaky_relu) #add activation if needed
```

forward pass
```python
new_node_features, new_edge_features = egat(sample_graph, node_features, edge_features)
```