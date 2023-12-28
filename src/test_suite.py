import unittest
import gc
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import torch
import networkx as nx
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from functions.net_class import Net
from functions.model_trainer import train_gnn_model, find_best_threshold, make_weight_prediction, train_xgboost_model

class ModelTesting(unittest.TestCase):
    def setUp(self):
        print("\nSetting up resources for the test")
        self.num_features = 255
        self.hidden_dim = 64
        self.num_classes = 1

    def test_encoding(self):
        gnn_model = Net(self.num_features, self.hidden_dim, self.num_classes)
        gnn_model.load_state_dict(torch.load('models/best_model.pth'))
        with open('test_data/unit_testing_graph_data', 'rb') as f:
            loaded_data = pickle.load(f)
            result = gnn_model.encode(loaded_data.x, loaded_data.edge_index)
        # Ensure that encoding's output shape is [NUM_NODES, NUM_CLASSES = 1]
        self.assertEqual(result.shape, torch.Size([len(loaded_data.x), self.num_classes]))

    def test_decoding(self):
        gnn_model = Net(self.num_features, self.hidden_dim, self.num_classes)
        gnn_model.load_state_dict(torch.load('models/best_model.pth'))
        with open('test_data/unit_testing_graph_data', 'rb') as f:
            loaded_data = pickle.load(f)
            z = gnn_model.encode(loaded_data.x, loaded_data.edge_index)
            neg_edge_index = negative_sampling(edge_index=loaded_data.edge_index, num_nodes=loaded_data.num_nodes,
                                           num_neg_samples=None, method='sparse')
            edge_label_index = torch.cat([loaded_data.edge_index, neg_edge_index], dim=-1, )
            edge_label = torch.cat([torch.ones(loaded_data.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))],
                                dim=0)
            # Ensure that edge_label has shape of RNS's shape + NUM_EDGES
            self.assertEqual(edge_label.shape[0], loaded_data.edge_index.shape[1] * 2)
            
            result = gnn_model.decode(z, edge_label_index).view(-1)

            # Ensure that predictions and labels are compatible
            self.assertEqual(result.shape[0], edge_label.shape[0])
    
    def test_RNS(self):
        with open('test_data/unit_testing_graph_data', 'rb') as f:
            loaded_data = pickle.load(f)
            neg_edge_index = negative_sampling(edge_index=loaded_data.edge_index, num_nodes=loaded_data.num_nodes,
                                            num_neg_samples=None, method='sparse')
            # Ensure that random negative sampling returned torch of shape [(Origin, Destination), NUM_EDGES]
            self.assertEqual(neg_edge_index.shape, torch.Size([2, loaded_data.edge_index.shape[1]]))
            # Ensure that no random negative sampled edge is a positive one
            self.assertEqual(((neg_edge_index[0] == loaded_data.edge_index[0]) & (neg_edge_index[1] == loaded_data.edge_index[1]))\
                             .nonzero().sum().item(), 0)

    def test_random_input_handling(self):
        gnn_model = Net(5, self.hidden_dim, self.num_classes)
        # 10 nodes, 30% probability of edge between each pair of nodes
        graph = nx.erdos_renyi_graph(10, p=0.3)
        edge_index = torch.tensor(list(graph.edges)).t().contiguous()
        # 10 nodes, 5 features
        x = torch.randn(10, 5)
        random_data = Data(x=x, edge_index=edge_index)
        train_gnn_model(gnn_model, random_data, 5)

        z = gnn_model.encode(random_data.x, random_data.edge_index)
        self.assertIsNotNone(z)
        self.assertFalse(torch.isnan(z).any())

        neg_edge_index = negative_sampling(edge_index = random_data.edge_index, num_nodes = None, \
                                           num_neg_samples = None, method = 'sparse')
        
        out = gnn_model.decode(z, neg_edge_index)
        self.assertIsNotNone(out)
        self.assertFalse(torch.isnan(out).any())

    def test_gnn_model_accuracy(self):
        gnn_model = Net(self.num_features, self.hidden_dim, self.num_classes)
        gnn_model.load_state_dict(torch.load('models/best_model.pth'))
        with open('test_data/unit_testing_graph_data', 'rb') as f:
            loaded_data = pickle.load(f)
            z = gnn_model.encode(loaded_data.x, loaded_data.edge_index)
            neg_edge_index = negative_sampling(edge_index=loaded_data.edge_index, num_nodes=loaded_data.num_nodes,
                                           num_neg_samples=None, method='sparse')
            edge_label_index = torch.cat([loaded_data.edge_index, neg_edge_index], dim=-1, )
            edge_label = torch.cat([torch.ones(loaded_data.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))],
                                dim=0)
            result = gnn_model.decode(z, edge_label_index).view(-1)
            threshold, error = find_best_threshold(result, edge_label)

            # Ensure at least 70% accuracy on binary edge predictions
            self.assertLessEqual(error, 0.3)
    
    def test_xgboost_accuracy(self):
        # 50 nodes, 30 features
        x = np.random.rand(50, 30)

        # Set the first column to be the row indices
        x[:, 0] = np.arange(50)

        # Convert values in the second to the last columns to 0 or 1
        x[:, 1:] = np.random.choice([0, 1], size=(50, 29))

        # Generate random weights matrix
        original_weights_matrix = np.random.rand(50, 30)

        train_xgboost_model(x, original_weights_matrix)
        xgboost_model = xgb.Booster()
        xgboost_model.load_model('models/xgboost_model_unit_testing.json')
        resulting_weights, mean_error = make_weight_prediction(0, original_weights_matrix, xgboost_model, \
                                                                np.array([original_weights_matrix[0]]), False, True)
        
        # Ensure at least 95% accuracy on edges weights predictions
        self.assertGreaterEqual(1-mean_error, 0.95)

    def tearDown(self):
        print("\nCleaning up resources after the test")
        del self.num_features
        del self.num_classes
        del self.hidden_dim
        gc.collect()

def layer_test():
    suite = unittest.TestSuite()
    suite.addTest(ModelTesting('test_encoding'))
    suite.addTest(ModelTesting('test_RNS'))
    suite.addTest(ModelTesting('test_decoding'))
    return suite

def compatibility_test():
    suite = unittest.TestSuite()
    suite.addTest('test_random_input_handling')
    return suite

def accuracy_test():
    suite = unittest.TestSuite()
    suite.addTest('test_gnn_model_accuracy')
    suite.addTest('test_xgboost_accuracy')
    return suite

if __name__ == '__main__':
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('.')

    runner = unittest.TextTestRunner()
    runner.run(test_suite)