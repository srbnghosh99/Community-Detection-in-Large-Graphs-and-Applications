import networkx as nx
import seaborn as sns
from pathlib import Path
import csv
import matplotlib.pyplot as plt  # Optional, for plotting
import ast
import argparse
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import json
from networkx.readwrite import json_graph
import numpy as np
import itertools
import random
import statistics
from sklearn.metrics import accuracy_score

column_names = ['Column1','Column2']
G = nx.read_edgelist('/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/renumbered_graph_ciao.csv',delimiter=' ', nodetype=int)
#ciao = pd.read_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/renumbered_graph_ciao.csv",sep = ' ',names=column_names)
#louvain = pd.read_csv('/Users/shrabanighosh/Downloads/data/trust_prediction/community_clusters/renumbered_graph_ciao_label_prop.csv',sep= ',')
cd_algo = pd.read_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/label_propagation_ciao_trustnet.csv")
cc = pd.read_csv('/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/propensity_subgraph/centerclusters.csv')
rating = pd.read_csv('/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/ciao_rating.csv')
# print(ciao,cc,rating)
# print()
# print(louvain)


# Create lists to store ground truth and predicted values
#ground_truth = []
#predicted_values = []

user_pairs = list(itertools.combinations(G.nodes(), 2))
#user_pairs = [e for e in G.edges]
#user_pairs = list(nx.all_pairs(G))
#print(user_pairs)

# Shuffle the user pairs randomly
random.shuffle(user_pairs)

# Calculate the size for the 'N' set (e.g., 50% of the pairs)
N_size = int(0.50 * len(user_pairs))
#
## Take the first N_size pairs for setting trust values to 0
N = user_pairs[:N_size]
print('pairs', len(N))
#
## Set trust values to 0 for pairs in 'N'
#for i, j in user_pairs[N_size:]:
#    G[i][j]['trust_value'] = 0
#
## Set trust values to 1 for the remaining pairs
#for i, j in N:
#    G[i][j]['trust_value'] = 1
#
#
#pairs_with_zero_trust = [(i, j) for i, j, data in G.edges(data=True) if data.get('trust_value', 0) == 0]
#for i, j in G.edges():
#    ground_truth.append(G[i][j]['trust_value'])
    
cols=['Node1', 'Node2', 'TrustValue']
lst = []
for i, j in G.edges():
    lst.append([i, j, 1])
ground_truth = pd.DataFrame(lst, columns=cols)


# Print the pairs with trust value 0
#for i, j in pairs_with_zero_trust:
#    print(f"Pair with trust value 0: ({i}, {j})")
    
    


#try:
#    df = pd.read_csv(file_path, sep=',')
#    print('File is comma-separated')
#except pd.errors.ParserError:
#    # If reading with comma as the delimiter fails, try space
#    try:
#        df = pd.read_csv(file_path, sep=' ')
#        print('File is space-separated')
#    except pd.errors.ParserError:
#        print('Delimiter could not be determined')
        
user_ratings = rating.groupby('userid')['rating'].agg(list).reset_index()
user_ratings['rating_vector'] = user_ratings['rating'].apply(np.array)

print(user_ratings)
print(user_ratings[user_ratings['userid'] == 1362])



# def calculate_similarity(user_vector, center_vector):
#     return np.dot(user_vector, center_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(center_vector))

def calculate_rating_similarity(rating_vector_i, rating_vector_j):
    len_i = len(rating_vector_i)
    len_j = len(rating_vector_j)

    max_len = max(len_i, len_j)

    # Zero-pad the shorter vectors to make them equal in length
    rating_vector_i = np.pad(rating_vector_i, (0, max_len - len_i))
    rating_vector_j = np.pad(rating_vector_j, (0, max_len - len_j))

    dot_product = np.sum(rating_vector_i * rating_vector_j)
    norm_i = np.sqrt(np.sum(rating_vector_i**2))
    norm_j = np.sqrt(np.sum(rating_vector_j**2))

    # Check for zero norm to avoid division by zero
    if norm_i == 0 or norm_j == 0:
        raise ValueError("Vector norms must be non-zero for similarity calculation.")
    
    rating_similarity = dot_product / (norm_i * norm_j)
    return rating_similarity


lst = []

centrality_measure_list = cc.colums
for cmeasure in centrality_measure_list:
    for i, j in N:
    #    i = row['Column1']
    #    j = row['Column2']
    #    print(i,j)
        comm_id_i = cd_algo[cd_algo['Node'] == i]['Community'].iloc[0]
    #    print('cluster number', comm_id_i)
        representative_node_of_i = cc[cc['Cluster'] == comm_id_i][cmeasure].iloc[0]
    #    print('representative_node_of_i',representative_node_of_i)
        comm_id_j = cd_algo[cd_algo['Node'] == i]['Community'].iloc[0]
        representative_node_of_j = cc[cc['Cluster'] == comm_id_j][cmeasure].iloc[0]
        user_vector = user_ratings[user_ratings['userid'] == i]['rating_vector'].iloc[0]
        center_vector_i = user_ratings[user_ratings['userid'] == representative_node_of_i]['rating_vector'].iloc[0]
        
        Rici = calculate_rating_similarity(user_vector,center_vector_i)
        user_vector = user_ratings[user_ratings['userid'] == j]['rating_vector'].iloc[0]
        center_vector_j = user_ratings[user_ratings['userid'] == representative_node_of_j]['rating_vector'].iloc[0]
        Rjcj = calculate_rating_similarity(user_vector,center_vector_j)
        CiCj = calculate_rating_similarity(center_vector_i,center_vector_j)

        # print(i,j)
    #    print(representative_node_of_i,representative_node_of_j,CiCj)
        # this one for non overlapping cluster. For ovelapping sum up with other cluster center as well.
        valuelist = [Rici,Rjcj,CiCj]
    #    print(statistics.mean(valuelist))
    #    predicted_values
        

        predicted_value = (statistics.mean([Rici, Rjcj, CiCj]))
        lst.append([i,j,predicted_value])

        # Append ground truth and predicted values
    #    ground_truth.append(G[i][j]['trust_value'])
    #    predicted_values.append(predicted_value)
        
    # Convert predicted values to binary (0 or 1)

    predicted_values = pd.DataFrame(lst, columns=cols)
    print(predicted_values)
    predicted_values['TrustValue'] = predicted_values['TrustValue'].apply(lambda avg: 1 if avg > 0.60 else 0)
    #predicted_values.to_csv("predicted_values.csv")
    #ground_truth.to_csv("ground_truth.csv")

    common_pairs = pd.merge(ground_truth, predicted_values, on=['Node1', 'Node2'], how='inner')
    print(common_pairs.shape)
    # Extract ground truth and predicted values for common pairs
    ground_truth_common = common_pairs['TrustValue_x'].tolist()
    predicted_values_common = common_pairs['TrustValue_y'].tolist()

    # Calculate accuracy for common pairs
    accuracy_common = accuracy_score(ground_truth_common, predicted_values_common)
    print("Centrality Measure", cmeasure)
    print("Accuracy for Common Pairs:", accuracy_common)


#predicted_values_binary = [1 if avg > 0.86 else 0 for avg in predicted_values]

    
#    # Assuming you have a function predict_trust_value(i, j) that predicts trust values
#    predicted_values = [predict_trust_value(i, j) for i, j in user_pairs]
#
#    # Extract actual trust values
#    actual_values = [G[i][j]['trust_value'] for i, j in user_pairs]
#
#    # Compare predictions with ground truth
#    correct_predictions = sum(predicted == actual for predicted, actual in zip(predicted_values, actual_values))
#
#    # Calculate accuracy
#    accuracy = correct_predictions / len(user_pairs)
#
#    print("Accuracy:", accuracy)


    # break 
    





'''

# Calculate the rating similarities for each user with their corresponding community centers
user_community_similarities = {}

for user, ratings in user_ratings.items():
    user_community_similarities[user] = {}
    for label in user_labels[user]:
        center = community_centers[label]
        similarity = calculate_similarity(ratings, center)
        user_community_similarities[user][label] = similarity

# Calculate the rating similarities between all community centers
community_similarity_matrix = np.zeros((len(community_centers), len(community_centers)))

for i, label_i in enumerate(community_centers):
    for j, label_j in enumerate(community_centers):
        center_i = community_centers[label_i]
        center_j = community_centers[label_j]
        similarity = calculate_similarity(center_i, center_j)
        community_similarity_matrix[i, j] = similarity

# Calculate the final trust value between users as the maximum overlap of community similarities
def calculate_final_trust(user1, user2):
    user1_similarities = user_community_similarities[user1]
    user2_similarities = user_community_similarities[user2]
    
    max_overlap = 0
    for label in user_labels[user1]:
        if label in user_labels[user2]:
            overlap = min(user1_similarities[label], user2_similarities[label])
            max_overlap = max(max_overlap, overlap)
    
    return max_overlap

# Example usage
user1 = 'user_i'
user2 = 'user_j'
final_trust_value = calculate_final_trust(user1, user2)
for index, row in ciao.iterrows():
    i = row['Column1']
    comm_id = louvain[louvain['Node'] == i]['Community'].iloc[0]
    
'''
