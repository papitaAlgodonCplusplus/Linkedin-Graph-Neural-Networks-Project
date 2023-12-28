import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
import plotly.graph_objects as go
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_gnn_model(gnn_model, data, epochs, plot=False, batched = False, name='models/gnn_model_unit_testing.pth'):
    """
    Trains a Graph Neural Network (GNN) model using the provided data.

    Args:
        gnn_model (torch.nn.Module): The GNN model to be trained.
        data (torch_geometric.data.Data or torch_geometric.data.Batch): The input graph data.
        epochs (int): The number of training epochs.
        plot (bool, optional): If True, a loss curve plot will be displayed using Plotly. Default is False.
        batched (bool, optional): If True, assumes data is a batch of graphs; otherwise, data is a single graph.
                                 Default is False.
        name (str, optional): The name of the file to save the trained model. Default is 'trained_model.pth'.

    Returns:
        None: The function modifies the provided GNN model in-place.

    Example:
        >>> train_gnn_model(my_gnn_model, my_graph_data, epochs=50, plot=True, batched=False)
    """
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    min_loss = float('inf')

    # Store loss values for plotting
    losses = []

    if not batched:
        for epoch in range(epochs):
            gnn_model.train()

            # Forward pass
            optimizer.zero_grad()
            z = gnn_model.encode(data.x, data.edge_index)
            neg_edge_index = negative_sampling(edge_index=data.edge_index, num_nodes=data.num_nodes,
                                            num_neg_samples=None, method='sparse')
            edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=-1, )
            edge_label = torch.cat([torch.ones(data.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))],
                                dim=0)
            out = gnn_model.decode(z, edge_label_index).view(-1)
            loss = criterion(out, edge_label)
            loss.backward()
            optimizer.step()
            if loss.item() < min_loss:
                min_loss = loss.item()
                torch.save(gnn_model.state_dict(), name)
            
            # Append current loss to the list
            losses.append(loss.item())
            if plot:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
        
        # Plotting the loss curve using Plotly
        if plot:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, epochs + 1)), y=losses, mode='lines+markers'))
            fig.update_layout(title='Loss Over Epochs', xaxis_title='Epoch', yaxis_title='Loss')
            fig.show()
        
    else:
        for epoch in range(epochs):
            gnn_model.train()

            for i in len(data):
                # Forward pass
                optimizer.zero_grad()
                z = gnn_model.encode(data[i].x, data[i].edge_index)
                neg_edge_index = negative_sampling(edge_index=data[i].edge_index, num_nodes=data[i].num_nodes,
                                                num_neg_samples=None, method='sparse')
                edge_label_index = torch.cat([data[i].edge_index, neg_edge_index], dim=-1, )
                edge_label = torch.cat([torch.ones(data[i].edge_index.size(1)), torch.zeros(neg_edge_index.size(1))],
                                    dim=0)
                out = gnn_model.decode(z, edge_label_index).view(-1)
                loss = criterion(out, edge_label)
                loss.backward()
                optimizer.step()
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    torch.save(gnn_model.state_dict(), name)

                # Append current loss to the list
                losses.append(loss.item())
                if plot:
                   print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

        # Plotting the loss curve using Plotly
        if plot:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, epochs + 1)), y=losses, mode='lines+markers'))
            fig.update_layout(title='Loss Over Epochs', xaxis_title='Epoch', yaxis_title='Loss')
            fig.show()

def train_xgboost_model(x, original_weights_matrix, name='models/xgboost_model_unit_testing.json'):
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, original_weights_matrix, test_size=0.2, random_state=42)

    # Convert the data into DMatrix format, which is the internal data structure used by XGBoost
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    # Define XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,  # Maximum depth of a tree.
        'learning_rate': 0.1,
    }

    model2 = xgb.train(params, dtrain)
    model2.save_model(name)

def make_weight_prediction(node_index, original_weights_matrix, xgboost_model , job, printing = False, return_mean_error = False):
  """
    Predicts weights for a specific node using an XGBoost model.

    Args:
        node_index (int): Index of the node for which weights are to be predicted.
        original_weights_matrix (numpy.ndarray): Original weights matrix containing weights for all nodes.
        xgboost_model (xgboost.Booster): Trained XGBoost model for weight prediction.
        job (xgb.DMatrix or numpy.ndarray): New data for which weights are to be predicted.
                                             If not xgb.DMatrix, it will be converted to xgb.DMatrix.
        printing (bool, optional): If True, print additional information about the predictions. Default is False.

    Returns:
        numpy.ndarray: Predicted weights for the specified node.

    Example:
        >>> node_index = 1
        >>> original_weights_matrix = np.array([[0.5, 0.8, 0.2], [0.3, 0.6, 0.1]])
        >>> xgboost_model = xgb.Booster()
        >>> job = np.array([[0.7, 0.2, 0.9]])
        >>> predictions = make_weight_prediction(node_index, original_weights_matrix, xgboost_model, job, printing=True)
    """
  if not isinstance(job, xgb.DMatrix):
    job = xgb.DMatrix(job)
  predictions = xgboost_model.predict(job)
  if printing:
    print("Predicted output (Correlation of the new data for all existing nodes):", predictions, predictions.shape)
    print("Difference mean: ", np.mean(predictions - original_weights_matrix[node_index]))
  if return_mean_error:
    return predictions, np.mean(predictions - original_weights_matrix[node_index])
  return predictions

def find_best_threshold(out, edge_label):
  """
  Finds the best threshold for a negative/positive edges classification task.

  This function iterates over a range of thresholds and calculates the error
  for each threshold. The threshold that minimizes the error is then selected
  as the best threshold.

  Parameters:
    - out (torch.Tensor): Model output tensor.
    - edge_label (torch.Tensor): Ground truth edge labels.

  Returns:
  - Tuple (float, float): A tuple containing the best threshold and the corresponding
    error rate.

  Example:
  ```python
  best_threshold, error_rate = find_best_threshold()
  print(f"Best Threshold: {best_threshold}, Error Rate: {error_rate}")
  ```
  """
  threshold = 1
  error = 1
  for i in np.arange(0, 1, 0.05):
    mask = (out > (torch.mean(out))*i).float()
    new_error = torch.sum(mask != edge_label).item() / len(edge_label)
    if new_error < error:
      threshold = i
      error = new_error
  return round(i, 2), error