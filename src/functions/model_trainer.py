import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import xgboost as xgb
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import plotly.graph_objects as go
import time

def draw_graph(graph, num_nodes_to_plot):
    message = st.empty()
    message.text("Drawing graph, please wait one minute.")
    pos = nx.spring_layout(graph)
    nodes_to_plot = list(graph.nodes)[:num_nodes_to_plot]
    subgraph = graph.subgraph(nodes_to_plot)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        text=[],  # Added to include edge weights as text
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',  # Updated hoverinfo to display text
        mode='lines'
    )

    for edge in subgraph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        textposition='top center',  # Display text above the nodes
        marker=dict(size=10)
    )

    for node in subgraph.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([str(node)])  # Add the node number as text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    st.plotly_chart(fig)
    message.empty()
  
def remove_last_node(data):    
    # Number of nodes in the original graph
    num_nodes = data.x.size(0)

    # Index of the last node
    last_node_index = num_nodes - 1

    # Remove the last node from x
    data.x = data.x[:last_node_index]
    row_mask = (data.edge_index[0, :] >= 0) & (data.edge_index[0, :] < last_node_index) \
            & (data.edge_index[1, :] >= 0) & (data.edge_index[1, :] < last_node_index)

    # Update edge_index by removing edges connected to the last node
    data.edge_index = data.edge_index[:, row_mask]
    data.y = data.y[row_mask]
  
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

def make_weight_prediction(xgboost_model ,job, final_weights_matrix, printing = False):
  """
  Make weight predictions using an XGBoost model.

  Parameters:
  - xgboost_model (xgb.Booster): The trained XGBoost model used for predictions.
  - job (xgb.DMatrix or array-like): The input data for which weight predictions are to be made.
      If not already an xgb.DMatrix, it will be converted to one.
  - printing (bool, optional): If True, print additional information about the predictions.
      Default is False.

  Returns:
  - predictions (numpy.ndarray): The predicted output, representing the correlation
    of the new data for all existing nodes.

  Example:
  >>> model = xgb.Booster()  # Replace with your trained XGBoost model
  >>> input_data = np.array([[...]])  # Replace with your input data
  >>> predictions = make_weight_prediction(model, input_data, printing=True)
  """
  if not isinstance(job, xgb.DMatrix):
    job = xgb.DMatrix(job)
  predictions = xgboost_model.predict(job)
  if printing:
    print("Predicted output (Correlation of the new data for all existing nodes):", predictions, predictions.shape)
    print("Difference mean: ", np.mean(predictions - final_weights_matrix[51]))
  return predictions

def predictions_to_df(predictions):
  results = np.array(predictions[0])
  results = np.hstack((results,  np.array(predictions[1]).reshape(len(predictions[1]),1)))
  results = pd.DataFrame(results, columns=['Origin', 'Destination', 'Weight'])
  results['Weight %'] = round(results['Weight'] * 100,2)
  results.iloc[:,0] = results.iloc[:,0].astype(int)
  results.iloc[:,1] = results.iloc[:,1].astype(int)
  return results

def predict_edges_and_weights(threshold, xgboost_model, gnn_model, data, printing=False, origin=None, destination=None, streamlist_session = False, on_epochs = False):
  """
  Predicts potential edges and their weights using a combination of graph neural network (GNN) and XGBoost models.

  Parameters:
  - threshold (float): The threshold value to determine the presence of an edge based on the GNN predictions.
  - xgboost_model: The pre-trained XGBoost model for edge weight prediction.
  - gnn_model: The pre-trained GNN model for edge presence prediction.
  - data: The input graph data containing features (data.x), edge indices (data.edge_index), and other information.
  - printing (bool, optional): If True, print additional information during execution. Default is False.
  - origin (int, optional): If provided, checks for predicted edges originating from this node.
  - destination (int, optional): If provided, checks for predicted edges leading to this node.

  Returns:
  - pred_edges (list): List of predicted edges in the format [(node1, node2), ...].
  - pred_weights (list): List of predicted weights corresponding to each edge in pred_edges.

  If origin and destination are provided:
  - If a connection is found between the specified nodes, returns (pred_edges, pred_weights).
  - If no connection is found, prints a message and returns None.

  If origin and destination are not provided:
  - Returns (pred_edges, pred_weights).

  If no predicted edges are found, returns None.
  """
  pred_edges = []
  pred_weights = []
  z = gnn_model.encode(data.x, data.edge_index)
  neg_edge_index = negative_sampling(edge_index=data.edge_index, num_nodes=data.num_nodes,
                                      num_neg_samples=None, method='sparse')
  edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=-1, )
  edge_label = torch.cat([torch.ones(data.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))],
                          dim=0)
  out = gnn_model.decode(z, edge_label_index).view(-1)
  pred = ((out[out.shape[0]//2:] > torch.mean(out) * threshold).float()).cpu().numpy()
  if printing:
    edge_label = torch.cat([torch.ones(data.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))],
                           dim=0)
    print("Error = {z}".format(z=\
    torch.sum((out > torch.mean(out)*threshold).float() != edge_label).item() / len(edge_label)))
  found = np.argwhere(pred == 1)
  if printing:
    print("Found: {z} possible edges".format(z=found.size))
  if found.size > 0:
      edge_tuples = edge_label_index.t().cpu().numpy()
      select_index = found.reshape(1, found.size)[0]
      pred_edges += edge_tuples[select_index].tolist()

  if not streamlist_session:
     for count, _ in enumerate(tqdm(pred_edges, desc="Processing Edges", unit="edge")):
      weights_matrix = make_weight_prediction(xgboost_model, np.array(data.x[pred_edges[count][0]])[np.newaxis, :],\
                                              printing)
      edge_weight = list(weights_matrix)[0][pred_edges[count][1]]
      pred_weights.append(edge_weight)
  else:
     subheader = st.subheader("Making Predictions for all Nodes in the Graph")
     time_msg = st.text(f"(Estimated Processing Time: {round((len(pred_edges)*0.01)/60, 1)} minutes)")
     start_time = time.time()

     # Create a Streamlit progress bar
     progress_bar = st.progress(0)

     for count, _ in enumerate(pred_edges):
       weights_matrix = make_weight_prediction(xgboost_model, np.array(data.x[pred_edges[count][0]])[np.newaxis, :],\
                                              printing)
       edge_weight = list(weights_matrix)[0][pred_edges[count][1]]
       pred_weights.append(edge_weight)
       progress_percentage = min((count + 1) / len(pred_edges), 1.0)
       progress_bar.progress(progress_percentage)
     
     if not on_epochs:
       st.success("Processing complete!")
       end_time = time.time()
       st.write(f"Time taken: {round((end_time - start_time)/60, 2)} minutes")
     else:
        subheader.empty()
        time_msg.empty()
        progress_bar.empty()
  if origin is not None and destination is not None:
      pred_edges = np.array(pred_edges)
      index = np.where((pred_edges[:, :] == origin) & (pred_edges[:, :] == destination))
      if index[0].size == 0:
          print("Model predicted no connection between {i}, {j}".format(i=origin, j=destination))
      else:
          print("Found connection between {i}, {j} at {k}".format(i=origin, j=destination, k=index[0]))
      return pred_edges, pred_weights
  else:
      return pred_edges, pred_weights

  return None

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