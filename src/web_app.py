import streamlit as st
import unittest
import gc
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import torch
import torch_geometric
import torch.nn.functional as F
import torch_geometric.transforms as T
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import random
import itertools
import plotly.graph_objects as go
import gc

from functions.net_class import Net
from functions.model_trainer import predict_edges_and_weights, find_best_threshold, predictions_to_df, remove_last_node, draw_graph
from functions.streamlit_lists import location_list, company_location_list, company_countries, job_benefits, job_skill_types, company_industries, validate_and_convert_data, data_types

st.write("""
# Linkedin Jobs Grouping App

This app takes LinkedIn job offers and predicts the similarity of each job with respect to the other jobs.
         
It allows you to either submit your own csv file with [the requested format](https://drive.google.com/file/d/1LeAFOqS_72kbP4y4ob6bmIO_eUwn-xou/view?usp=sharing) so the AI will
predict new edges and assign weights (similarity) to it based on the [predefined jobs file](https://raw.githubusercontent.com/papitaAlgodonCplusplus/Misc/main/example_jobs.csv), creating new connections between jobs as it predicts likeness between them,
or, you can also manually input a new job from the sidebar at the left, and the model will use the [predefined jobs file](https://raw.githubusercontent.com/papitaAlgodonCplusplus/Misc/main/example_jobs.csv) as
the neighbors of the new given job that will have predicted connections and weights (similarity).
         
*Data obtained from the [Linkedin Jobs Dataseet](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) by ARSH KON.*
""")

st.sidebar.header('New Job Features')

st.sidebar.markdown("""
[Example CSV input file](https://drive.google.com/file/d/1LeAFOqS_72kbP4y4ob6bmIO_eUwn-xou/view?usp=sharing)
""")

# Load the company_list from the pickle file
with open("test_data/company_city_list.pkl", 'rb') as file:
    company_city_list = pickle.load(file)

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    epochs = st.number_input("**Fitting Epochs (More Epochs, More Connections)**", value=3, step=1)
    input_data = pd.read_csv(uploaded_file)
    input_data_checkpoint = input_data
    node_features = pd.read_csv('https://git.ucr.ac.cr/ALEXANDER.QUESADAQUESADA/misc/-/raw/main/example_jobs_original.csv')
    title_dummies_filtered = pd.read_csv('test_data/title_dummies_filtered.csv')
    # Parse new data with downloaded parsers
    input_data = pd.merge(input_data, title_dummies_filtered, left_index=True, right_index=True, how='left')
    input_data.drop(columns='title', inplace=True)
    del title_dummies_filtered

    # Since it's impossible to summarize a whole job description into a reasonable amount of columns
    # Nor encode them in a numerical way using complex NLP (Transfomer based) for time reasons
    # We won't use this column for features
    input_data.drop(columns='job_desc', inplace=True)
    # Same applies for company's description
    input_data.drop(columns='company_desc', inplace=True)
    # Also, since we have company's id, company's name is obsolete
    input_data.drop(columns='company_name', inplace=True)

    # Work Type
    unique_work_types = node_features['formatted_work_type'].unique()
    mapping = {work_type: i for i, work_type in enumerate(unique_work_types)}

    input_data['formatted_work_type'] = input_data['formatted_work_type'].map(mapping)

    # Location
    input_data[['city', 'state']] = input_data['location'].str.split(', ', n=1, expand=True)
    node_features[['city', 'state']] = node_features['location'].str.split(', ', n=1, expand=True)
    node_features.drop(columns='location', inplace=True)
    input_data.drop(columns='location', inplace=True)

    uniques = node_features['city'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    input_data['city'] = input_data['city'].map(mapping)

    # Company Country
    uniques = node_features['company_country'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    input_data['company_country'] = input_data['company_country'].map(mapping)

    # Experience Level
    uniques = node_features['formatted_experience_level'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    input_data['formatted_experience_level'] = input_data['formatted_experience_level'].map(mapping)

    # Company State
    uniques = node_features['company_state'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    input_data['company_state'] = input_data['company_state'].map(mapping)

    uniques = node_features['state'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    input_data['state'] = input_data['state'].map(mapping)

    # Application Type
    uniques = node_features['application_type'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    input_data['application_type'] = input_data['application_type'].map(mapping)

    # Job Required Skill
    uniques = node_features['job_skill_type'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    input_data['job_skill_type'] = input_data['job_skill_type'].map(mapping)

    # Job Benefit
    uniques = node_features['job_benefit'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    input_data['job_benefit'] = input_data['job_benefit'].map(mapping)

    # Company Size
    input_data['company_size'] = input_data['company_size'].astype(int)

    # Company's city
    uniques = node_features['company_city'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    input_data['company_city'] = input_data['company_city'].map(mapping)

    # Company's Industry
    uniques = node_features["company's industry"].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    input_data["company's industry"] = input_data["company's industry"].map(mapping)

    # Company's Roles
    def remove_duplicates(row):
        roles = row.split(', ')
        unique_roles = list(set(roles))
        return ', '.join(unique_roles)

    input_data["company's roles"].fillna("None", inplace=True)
    input_data["company's roles"] = input_data["company's roles"].apply(remove_duplicates)

    uniques = node_features["company's roles"].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    input_data["company's roles"] = input_data["company's roles"].map(mapping)

    # Drop redundant info
    input_data.drop(columns='company_address', inplace=True)
    input_data.drop(columns='Unnamed: 0_x', inplace=True)
    input_data.drop(columns='Unnamed: 0_y', inplace=True)
    def convert_to_float_or_zero(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0
    input_data['company_zip_code'] = input_data['company_zip_code'].apply(convert_to_float_or_zero)
    del uniques
    del mapping
    x = torch.from_numpy(input_data.to_numpy().astype(float)).float().squeeze()

    adjacency_matrix_job_types = np.zeros((x.shape[0], x.shape[0]), dtype=np.int32)

    job_types = input_data_checkpoint['job_skill_type'].unique()

    for i in job_types:
        df_by_job_type_temp = input_data_checkpoint[input_data_checkpoint['job_skill_type'] == i]
        row_indexes = df_by_job_type_temp.index.to_numpy()
        adjacency_matrix_job_types[row_indexes[:, np.newaxis], row_indexes] = 1
    adjacency_matrix_job_types[np.arange(x.shape[0]), np.arange(x.shape[0])] = 0

    del job_types
    gc.collect()

    title_sub_df = input_data.loc[:, "company's industry": "city"]
    title_sub_df.drop(columns="company's industry", inplace=True)
    title_sub_df.drop(columns="city", inplace=True)
    adjacency_matrix_title = np.zeros((len(title_sub_df.columns), x.shape[0], x.shape[0]), dtype=float)
    for index, column_name in enumerate(title_sub_df.columns):
        df_by_title_temp = input_data[input_data[column_name] == 1]
        row_indexes = df_by_title_temp.index.to_numpy()
        adjacency_matrix_title[index, row_indexes[:, np.newaxis], row_indexes] = 1

    del df_by_title_temp
    del df_by_job_type_temp
    gc.collect()

    n = len(title_sub_df.columns)
    for i in range(len(title_sub_df.columns)):
        adjacency_matrix_title[n-1] = \
        adjacency_matrix_title[n-1] + adjacency_matrix_title[i]

    adjacency_matrix_title = adjacency_matrix_title[n-1]
    adjacency_matrix_title[np.arange(x.shape[0]), np.arange(x.shape[0])] = 0

    title_sub_df['Sum'] = title_sub_df.sum(axis=1)
    sum_array = title_sub_df['Sum'].to_numpy()
    del title_sub_df
    gc.collect()
    adjacency_matrix_title /= np.maximum.outer(sum_array, sum_array)
    adjacency_matrix_title = np.round(adjacency_matrix_title, decimals=2)
    adjacency_matrix_title = np.nan_to_num(adjacency_matrix_title, nan=0)

    adjacency_matrix_company_industry = np.zeros((x.shape[0], x.shape[0]), dtype=np.int32)
    industry_types = node_features["company's industry"].unique()
    for i in industry_types:
        df_by_industry_temp = input_data[input_data["company's industry"] == i]
        row_indexes = df_by_industry_temp.index.to_numpy()
        adjacency_matrix_company_industry[row_indexes[:, np.newaxis], row_indexes] = 1

    adjacency_matrix_company_industry[np.arange(x.shape[0]), np.arange(x.shape[0])] = 0

    adjacency_matrix_company_roles = np.zeros((x.shape[0], x.shape[0]), dtype=np.int32)

    roles_types = input_data["company's roles"].unique()

    for i in roles_types:
        df_by_roles_temp = input_data[input_data["company's roles"] == i]
        row_indexes = df_by_roles_temp.index.to_numpy()
        adjacency_matrix_company_roles[row_indexes[:, np.newaxis], row_indexes] = 1
    adjacency_matrix_company_roles[np.arange(x.shape[0]), np.arange(x.shape[0])] = 0

    final_weights_matrix = np.zeros((x.shape[0], x.shape[0]), dtype=float)
    final_weights_matrix = (adjacency_matrix_job_types  * 0.25) + ((adjacency_matrix_title/2) * 0.65) \
    + (adjacency_matrix_company_industry * 0.05) + (adjacency_matrix_company_roles * 0.05)

    del adjacency_matrix_job_types
    del adjacency_matrix_title
    del adjacency_matrix_company_industry
    del adjacency_matrix_company_roles
    gc.collect()

    final_weights_matrix = np.round(final_weights_matrix, decimals=8)
    final_weights_matrix = np.nan_to_num(final_weights_matrix, nan=0)
    final_weights_matrix[np.arange(x.shape[0]), np.arange(x.shape[0])] = 0  
    edges_matrix = (final_weights_matrix != 0).astype(int)
    repeated_indices = []
    edges_indices = []

    for i in range(x.shape[0]):
        row = edges_matrix[i]
        ones_indices = np.where(row == 1)[0]
        repeated_indices.extend([i] * len(ones_indices))
        edges_indices.extend(ones_indices)

    result_array = np.array(repeated_indices)
    edges_array = np.array(edges_indices)

    del repeated_indices
    del edges_indices
    gc.collect()
    all_edges = np.array((result_array, edges_array), dtype = np.int32)
    labels = final_weights_matrix[final_weights_matrix != 0.0].flatten()
    x = torch.where(torch.isnan(x), torch.tensor(0.0), x)

    non_numeric_values = {}
    numeric_value = 0

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            value = x[i, j]
            if not np.issubdtype(type(value), np.number):
                if value not in non_numeric_values:
                    non_numeric_values[value] = numeric_value
                    numeric_value += 1
                x[i, j] = non_numeric_values[value]
    
    loaded_data = Data(x=x, edge_index=torch.from_numpy(all_edges), y=torch.from_numpy(labels))
    gnn_model = Net(255, 64, 1)
    gnn_model.load_state_dict(torch.load('models/best_model.pth'))

    xgboost_model = xgb.Booster()
    xgboost_model.load_model('models/xgboost_model.model')

    z = gnn_model.encode(loaded_data.x, loaded_data.edge_index)
    neg_edge_index = negative_sampling(edge_index=loaded_data.edge_index, num_nodes=loaded_data.num_nodes,
                                    num_neg_samples=None, method='sparse')
    edge_label_index = torch.cat([loaded_data.edge_index, neg_edge_index], dim=-1, )
    edge_label = torch.cat([torch.ones(loaded_data.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))],
                        dim=0)
    result = gnn_model.decode(z, edge_label_index).view(-1)
    threshold, error = find_best_threshold(result, edge_label)

    if st.button("Make Predictions"):
        # Centered text with color and box
        centered_text = f"""<div style="text-align: center; padding: 10px; border: 2px solid #3498db; color: #3498db; background-color: #ecf0f1; border-radius: 5px;">
                    Processing Graph with {len(loaded_data.x)} Nodes  
                    </div>"""
        # Display the centered text
        st.markdown(centered_text, unsafe_allow_html=True)
        st.markdown("---")
        for i in range(0, epochs):
            centered_text2 = f"""<div style="text-align: center; padding: 10px; border: 2px solid #1f9c1f; color: #188018; background-color: #cafaca; border-radius: 5px;">
                    Epoch {i}/{epochs}
                    </div>"""
            epoch_msg = st.markdown(centered_text2, unsafe_allow_html=True)
            predictions = predict_edges_and_weights(threshold, xgboost_model, gnn_model, loaded_data, True, streamlist_session=True, on_epochs=True)
            results = predictions_to_df(predictions)
            new_edges = torch.tensor([results['Origin'].values, results['Destination'].values])
            loaded_data.edge_index = torch.cat((loaded_data.edge_index , new_edges), dim=1).to(torch.int64)
            if isinstance(loaded_data.y, torch.Tensor):
                loaded_data.y = torch.cat((loaded_data.y, torch.tensor(results['Weight'].values)), dim=0)
            else:
                loaded_data.y = torch.cat((torch.from_numpy(loaded_data.y), torch.tensor(results['Weight'].values)), dim=0)
            epoch_msg.empty()
        results['Origin'] = results['Origin'] + 1
        results['Destination'] = results['Destination'] + 1
        st.subheader('Predicted Edges and Weights')
        st.write(results)
        st.subheader('Your Input Data')
        st.write(input_data_checkpoint)

        def to_networkx(data):
            edge_index = data.edge_index.numpy()
            edge_weights = data.y.numpy()

            edge_list = [(edge_index[0, i], edge_index[1, i], edge_weights[i]) for i in range(edge_index.shape[1])]

            G = nx.Graph()
            G.add_nodes_from(range(data.num_nodes))
            G.add_weighted_edges_from(edge_list)

            return G
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("Network Visualization of the new Predicted Graph")
        draw_graph(to_networkx(loaded_data), len(loaded_data.x)-1)

else:
    def user_input_features():
        job_id = st.sidebar.number_input("Job ID", value=881656947, step=1)
        company_id = st.sidebar.number_input("Company ID", value=485896947, step=1)
        title = st.sidebar.text_input('Job Title', "Sales Manager")
        formatted_work_type = st.sidebar.selectbox("Job Type", ["Full-time", "Contract", "Part-time"])
        location = st.sidebar.selectbox("Location", location_list)
        applies = st.sidebar.number_input("Number of Applies", value=0, step=1)
        views = st.sidebar.number_input("Number of Views", value=30, step=1)
        application_type = st.sidebar.selectbox("Type of Application", ['ComplexOnsiteApply', 'OffsiteApply', 'SimpleOnsiteApply'])
        formatted_experience_level = st.sidebar.selectbox("Level of Experience", ['Mid-Senior level', 'Entry level', 'Associate',
                                                                                  'Director', 'Executive', 'Internship'])
        sponsored = st.sidebar.checkbox("Sponsored Job")
        company_size = st.sidebar.number_input("Company Size (0 Smallest - 7 Largest)", value=5, step=1)
        company_state = st.sidebar.selectbox("Company State", company_location_list)
        company_country = st.sidebar.selectbox("Company Country", company_countries)
        company_city = st.sidebar.selectbox("Company City", company_city_list)
        company_zip_code = st.sidebar.number_input("Company zip code", value=94103, step=1)
        company_address = st.sidebar.text_input('Company Address', "630 West 168th St.")
        job_benefit = st.sidebar.selectbox("Highlight Benefit of the Job", job_benefits)
        job_skill_type = st.sidebar.selectbox("Job Requiered Skill Type", job_skill_types)
        company_roles = st.sidebar.text_input('Roles of the Company', "Export & Import Clearance, Bonded Warehousing, Skilled Nursing, Physical Therapy")
        company_industry = st.sidebar.selectbox('Industry Type of the Company', company_industries)
        data = {'job_id': job_id,
                'company_id': company_id,
                'title': title,
                'job_desc': "na",
                'formatted_work_type': formatted_work_type,
                'location': location,
                'applies': applies,
                'views': views,
                'application_type': application_type,
                'formatted_experience_level': formatted_experience_level,
                'sponsored': int(sponsored),
                'company_name': 'na',
                'company_desc': 'na',
                'company_size': company_size,
                'company_state': company_state,
                'company_country': company_country,
                'company_city': company_city,   
                'company_zip_code': company_zip_code,
                'company_address': company_address,
                'job_benefit': job_benefit,
                'job_skill_type': job_skill_type,
                "company's roles": company_roles,
                "company's industry": company_industry}
        return data
    input_data = user_input_features()

    node_features_encoded = pd.read_csv('test_data/example_jobs.csv')
    node_features = pd.read_csv('https://git.ucr.ac.cr/ALEXANDER.QUESADAQUESADA/misc/-/raw/main/example_jobs_original.csv')
    title_dummies_filtered = pd.read_csv('test_data/title_dummies_filtered.csv')
    new_data = pd.DataFrame(input_data, index=[0])
    new_data_checkpoint = pd.DataFrame(input_data, index=[0])

    # Parse new data with downloaded parsers
    new_data = pd.merge(new_data, title_dummies_filtered, left_index=True, right_index=True, how='left')
    new_data.drop(columns='title', inplace=True)
    del title_dummies_filtered

    # Since it's impossible to summarize a whole job description into a reasonable amount of columns
    # Nor encode them in a numerical way using complex NLP (Transfomer based) for time reasons
    # We won't use this column for features
    new_data.drop(columns='job_desc', inplace=True)
    # Same applies for company's description
    new_data.drop(columns='company_desc', inplace=True)
    # Also, since we have company's id, company's name is obsolete
    new_data.drop(columns='company_name', inplace=True)

    # Work Type
    unique_work_types = node_features['formatted_work_type'].unique()
    mapping = {work_type: i for i, work_type in enumerate(unique_work_types)}

    new_data['formatted_work_type'] = new_data['formatted_work_type'].map(mapping)

    # Location
    new_data[['city', 'state']] = new_data['location'].str.split(', ', n=1, expand=True)
    node_features[['city', 'state']] = node_features['location'].str.split(', ', n=1, expand=True)
    node_features.drop(columns='location', inplace=True)
    new_data.drop(columns='location', inplace=True)

    uniques = node_features['city'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    new_data['city'] = new_data['city'].map(mapping)

    # Company Country
    uniques = node_features['company_country'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    new_data['company_country'] = new_data['company_country'].map(mapping)

    # Experience Level
    uniques = node_features['formatted_experience_level'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    new_data['formatted_experience_level'] = new_data['formatted_experience_level'].map(mapping)

    # Company State
    uniques = node_features['company_state'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    new_data['company_state'] = new_data['company_state'].map(mapping)

    uniques = node_features['state'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    new_data['state'] = new_data['state'].map(mapping)

    # Application Type
    uniques = node_features['application_type'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    new_data['application_type'] = new_data['application_type'].map(mapping)

    # Job Required Skill
    uniques = node_features['job_skill_type'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    new_data['job_skill_type'] = new_data['job_skill_type'].map(mapping)

    # Job Benefit
    uniques = node_features['job_benefit'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    new_data['job_benefit'] = new_data['job_benefit'].map(mapping)

    # Company Size
    new_data['company_size'] = new_data['company_size'].astype(int)

    # Company's city
    uniques = node_features['company_city'].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    new_data['company_city'] = new_data['company_city'].map(mapping)

    # Company's Industry
    uniques = node_features["company's industry"].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    new_data["company's industry"] = new_data["company's industry"].map(mapping)

    # Company's Roles
    def remove_duplicates(row):
        roles = row.split(', ')
        unique_roles = list(set(roles))
        return ', '.join(unique_roles)

    new_data["company's roles"] = new_data["company's roles"].apply(remove_duplicates)

    uniques = node_features["company's roles"].unique()
    mapping = {unique_value: i for i, unique_value in enumerate(uniques)}
    new_data["company's roles"] = new_data["company's roles"].map(mapping)

    # Drop redundant info
    new_data.drop(columns='company_address', inplace=True)
    new_data.drop(columns='job_id', inplace=True)
    new_data.drop(columns='Unnamed: 0', inplace=True)

    del uniques
    del mapping
    x = torch.from_numpy(new_data.to_numpy()).float().squeeze()

    gnn_model = Net(255, 64, 1)
    gnn_model.load_state_dict(torch.load('models/best_model.pth'))

    xgboost_model = xgb.Booster()
    xgboost_model.load_model('models/xgboost_model.model')

    with open('test_data/data_web_app', 'rb') as f:
        loaded_data = pickle.load(f)
        for value, _ in enumerate(x):
            if np.isnan(x[value]):
                x[value] = 0.0
        
        # Convert NumPy arrays to PyTorch tensors with compatible data types
        loaded_data.x  = torch.tensor(loaded_data.x.astype(np.float32), dtype=torch.float32)
        loaded_data.edge_index  = torch.tensor(loaded_data.edge_index, dtype=torch.long)   

        loaded_data.x = torch.cat([loaded_data.x, x.view(1, -1)], dim=0)
        z = gnn_model.encode(loaded_data.x, loaded_data.edge_index)
        neg_edge_index = negative_sampling(edge_index=loaded_data.edge_index, num_nodes=loaded_data.num_nodes,
                                        num_neg_samples=None, method='sparse')
        edge_label_index = torch.cat([loaded_data.edge_index, neg_edge_index], dim=-1, )
        edge_label = torch.cat([torch.ones(loaded_data.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))],
                            dim=0)
        result = gnn_model.decode(z, edge_label_index).view(-1)
        threshold, error = find_best_threshold(result, edge_label)

        if st.button("Make Predictions"):
            # Centered text with color and box
            centered_text = f"""<div style="text-align: center; padding: 10px; border: 2px solid #3498db; color: #3498db; background-color: #ecf0f1; border-radius: 5px;">
                            Processing Graph with {len(loaded_data.x)} Nodes  
                            </div>"""
            # Display the centered text
            st.markdown(centered_text, unsafe_allow_html=True)
            predictions = predict_edges_and_weights(threshold, xgboost_model, gnn_model, loaded_data, True, streamlist_session=True)
            results = predictions_to_df(predictions)
            results['Origin'] = results['Origin'] + 1
            results['Destination'] = results['Destination'] + 1
            st.subheader('Predicted Edges and Weights | All Nodes')
            st.write(results)
            st.subheader('Predicted Edges and Weights | New Input Node')
            st.write(results[(results['Origin'] == len(loaded_data.x)) | (results['Destination'] == len(loaded_data.x))])
            st.subheader('Information about All Nodes in the Graph')
            original_data = node_features.iloc[14500:14700].reset_index(drop=True)
            st.write(original_data.drop(original_data.columns[0], axis=1))
            st.subheader("Job Data from Left Sidebar")
            st.write(new_data_checkpoint)

            def to_networkx(data):
                edge_index = data.edge_index.numpy()
                edge_weights = data.y.numpy()

                edge_list = [(edge_index[0, i], edge_index[1, i], edge_weights[i]) for i in range(edge_index.shape[1])]

                G = nx.Graph()
                G.add_nodes_from(range(data.num_nodes))
                G.add_weighted_edges_from(edge_list)

                return G
            
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader("Network Visualization of the new Predicted Graph")
            new_edges = torch.tensor([results['Origin'].values, results['Destination'].values])
            loaded_data.edge_index = torch.cat((loaded_data.edge_index , new_edges), dim=1)
            loaded_data.y = torch.cat((torch.from_numpy(loaded_data.y), torch.tensor(results['Weight'].values)), dim=0)
            draw_graph(to_networkx(loaded_data), len(loaded_data.x)-1)
