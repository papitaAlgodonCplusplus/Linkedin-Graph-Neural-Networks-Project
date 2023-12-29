import gnnlens
import torch
import pickle
import shutil
import os
from torch_geometric.data import Data
from dgl import DGLGraph
from gnnlens import Writer
import numpy as numpy

with open('test_data/sample_gnnlens2_data', 'rb') as f:
    original_data = pickle.load(f)
original_data.edge_attr = torch.from_numpy(original_data.y)
original_data.y = None

with open('test_data/predicted_new_data', 'rb') as f:
    predicted_new_data = pickle.load(f)
predicted_new_data.edge_attr = predicted_new_data.y
predicted_new_data.y = None

# Function to convert torch_geometric Data to DGLGraph
def torch_geometric_to_dgl(data):
    dgl_graph = DGLGraph()
    dgl_graph.add_nodes(data.num_nodes)
    dgl_graph.add_edges(data.edge_index[0], data.edge_index[1])

    # Copy node features
    dgl_graph.ndata['x'] = data.x

    # Copy edge features if available
    if 'edge_attr' in data:
        dgl_graph.edata['edge_attr'] = data.edge_attr

    return dgl_graph

original_graph = torch_geometric_to_dgl(original_data)
predicted_graph = torch_geometric_to_dgl(predicted_new_data)

if os.path.exists("sample_gnnlens2_app"):
    try:
        shutil.rmtree("sample_gnnlens2_app")
    except Exception as e:
        print(f"Error: Unable to delete the folder 'sample_gnnlens2_app'.")
        print(e)

# Specify the path to create a new directory for dumping data files.
writer = Writer('sample_gnnlens2_app')
colors = torch.randint(0, 10, size=(len(original_data.x),))
colors2 = torch.randint(0, 10, size=(len(predicted_new_data.x),))
writer.add_graph(name='Sample_DGLGraph_Data', graph=original_graph, eweights={"edge_weights": \
                                                                         original_data.edge_attr.view(len(original_data.edge_attr))},
                                                                         nlabels= colors,
                                                                         num_nlabel_types = len(torch.unique(colors)))
writer.add_graph(name='Predicted_DGLGraph_Data', graph=predicted_graph, eweights={"edge_weights": \
                                                                         predicted_new_data.edge_attr.view(len(predicted_new_data.edge_attr))},
                                                                         nlabels= colors2,
                                                                         num_nlabel_types = len(torch.unique(colors2)))
writer.add_model(graph_name='Sample_DGLGraph_Data', model_name='NONE',
                 nlabels=colors)
writer.add_model(graph_name='Predicted_DGLGraph_Data', model_name='GNN_RNS_XGBOOST',
                 nlabels=colors2)
# Finish dumping
writer.close()
