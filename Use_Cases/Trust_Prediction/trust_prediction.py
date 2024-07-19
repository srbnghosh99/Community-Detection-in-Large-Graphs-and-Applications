import networkx as nx
import seaborn as sns
from pathlib import Path
import csv
import matplotlib.pyplot as plt
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
from tqdm import tqdm
import time
import os
from os.path import dirname, join as pjoin



def calculate_rating_similarity(rating_vector_i, rating_vector_j):
    len_i = len(rating_vector_i)
    len_j = len(rating_vector_j)
    max_len = max(len_i, len_j)
    # Compute the numerator as the dot product
    # common_keys = set(ratings_i.keys()).intersection(ratings_j.keys())
    # numerator = sum(ratings_i[k] * ratings_j[k] for k in common_keys)


    # numerator = sum(rating_vector_i[k] * rating_vector_j[k] for k in range(len(rating_vector_i)))
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

def calculate_rating_similarity2(rating_vector_i, rating_vector_j):
    comm_id_i = cd_algo[cd_algo['Node'] == i]['Community'].iloc[0]
    comm_id_j = cd_algo[cd_algo['Node'] == j]['Community'].iloc[0]
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



def prediction(dataset,graphfile,communityfile,community_center,ratingfile,overlap):
    curr_directory = os.getcwd()
    graphfile = pjoin(curr_directory,dataset, graphfile)
    communityfile = pjoin(curr_directory,dataset, communityfile)
    community_center = pjoin(curr_directory,dataset, community_center)
    ratingfile = pjoin(curr_directory,dataset, ratingfile)
    
    column_names = ['Column1','Column2']
    
    G = nx.read_edgelist(graphfile,delimiter=' ', nodetype=int)
    #G = pd.read_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/renumbered_graph_ciao.csv",sep = ' ',names=column_names)
    #louvain = pd.read_csv('/Users/shrabanighosh/Downloads/data/trust_prediction/community_clusters/renumbered_graph_ciao_label_prop.csv',sep= ',')
    cd_algo = pd.read_csv(communityfile,sep = ' ')
    cc = pd.read_csv(community_center)
    rating = pd.read_csv(ratingfile)
#    print(cd_algo, cc, ratingfile)
    print(ratingfile)

    x = [50,60,70,80,90]
    
    fname = pjoin(curr_directory,dataset, 'ground_truth.csv')
    df = pd.read_csv(fname)
    
    ground_truth = df.sample(frac=0.5, random_state=42)
            
    user_ratings = rating.groupby('userid')['rating'].agg(list).reset_index()
    user_ratings['rating_vector'] = user_ratings['rating'].apply(np.array)

#    print('cd_algo',cd_algo)


    lst = []
    centrality_measure_list = ['MaxClosenessNode','MaxSameAsDegreeCentralityNode','MaxBetweennessNode','MaxOutCentralityNode','MaxinCentralityNode','RandomNode']
    cols=['Node1', 'Node2', 'TrustValue']
    if (overlap == 'overlapping'):
        print(overlap)
        centrality_measure_list = ['MaxOutCentralityNode','MaxinCentralityNode','RandomNode']
        print(cd_algo['Node'].max())
        for cmeasure in centrality_measure_list:
            for index, row in tqdm(ground_truth.iterrows(), total=len(ground_truth)):
                i = row['Node1']
                j = row['Node2']
            # for i, j in tqdm(N,desc="Processing pairs"):
                comm_id_i = cd_algo[cd_algo['Node'] == i]['Community'].iloc[0]
                comm_id_j = cd_algo[cd_algo['Node'] == j]['Community'].iloc[0]
                # print(len(comm_id_i), len(comm_id_j))
                if (len(comm_id_i) == 2 or len(comm_id_j) == 2):   # some nodes not assigned to any community
                    continue
                comm_id_i = [int(x) for x in comm_id_i.strip('[]').split(', ')]
                comm_id_j = [int(x) for x in comm_id_j.strip('[]').split(', ')]
                avg_predicted_values = []
                # print('comm_id_i',{comm_id_i}, comm_id_j, {comm_id_j})
                # combinations = list(itertools.product(comm_id_i, comm_id_i))
                # print(len(comm_id_i), len(comm_id_j))
                # for id1, id2 in zip(comm_id_i, comm_id_j):
                # calculate RiCi (i to community centers i belongs to)
                user_vector_i = user_ratings[user_ratings['userid'] == i]['rating_vector'].iloc[0]
                user_vector_j = user_ratings[user_ratings['userid'] == j]['rating_vector'].iloc[0]
                sum = 0

                for c_i in comm_id_i:
                    for c_j in comm_id_j:
                        representative_node_c_i = cc[cc['Cluster'] == c_i][cmeasure].iloc[0]
                        representative_node_c_j = cc[cc['Cluster'] == c_j][cmeasure].iloc[0]
                        center_vector_c_i = user_ratings[user_ratings['userid'] == representative_node_c_i]['rating_vector'].iloc[0]
                        center_vector_c_j = user_ratings[user_ratings['userid'] == representative_node_c_j]['rating_vector'].iloc[0]
                        R_ic_i = calculate_rating_similarity(user_vector_i, center_vector_c_i)
                        R_jc_j = calculate_rating_similarity(user_vector_j, center_vector_c_j)
                        R_c_i_c_j = calculate_rating_similarity(center_vector_c_i, center_vector_c_j)
                        predicted_value = np.mean([R_ic_i, R_jc_j, R_c_i_c_j])
                        avg_predicted_values.append(predicted_value)


                max_predict = max(avg_predicted_values)
                lst.append([i,j,max_predict])

            predicted_values = pd.DataFrame(lst, columns=cols)
            print('predicted_values', predicted_values)
            predicted_values['TrustValue_new'] = predicted_values['TrustValue'].apply(lambda avg: 1 if avg > 0.80 else 0)
            
            common_pairs = pd.merge(ground_truth, predicted_values, on=['Node1', 'Node2'], how='inner')
            print(common_pairs.shape)
            common_pairs = common_pairs.rename(columns={'TrustValue_x': 'ground_truth', 'TrustValue_new': 'predicted_value','TrustValue_y':'score'})
            filename =  cmeasure + "_predict_ground_truth.csv"
            common_pairs.to_csv(filename)
            
            # Extract ground truth and predicted values for common pairs
            ground_truth_common = common_pairs['ground_truth'].tolist()
            predicted_values_common = common_pairs['predicted_value'].tolist()
          

            # Calculate accuracy for common pairs
            accuracy_common = accuracy_score(ground_truth_common, predicted_values_common)
            print("Centrality Measure", cmeasure)
            print("Accuracy for Common Pairs:", accuracy_common)
    
    else:
        print(overlap)
        print(cd_algo)
#        centrality_measure_list = ['MaxOutCentralityNode','MaxinCentralityNode','RandomNode']
        centrality_measure_list = ['MaxSameAsDegreeCentralityNode','MaxBetweennessNode','MaxOutCentralityNode','MaxinCentralityNode','RandomNode']
        # centrality_measure_list = ['MaxOutCentralityNode']
        print(cd_algo['Node'].max())
        for cmeasure in centrality_measure_list:
            for index, row in tqdm(ground_truth.iterrows(), total=len(ground_truth)):
                i = row['Node1']
                j = row['Node2']
            # for i, j in tqdm(N,desc="Processing pairs"):
                id1 = cd_algo[cd_algo['Node'] == i]['Community'].iloc[0]
                id2 = cd_algo[cd_algo['Node'] == j]['Community'].iloc[0]
                # print(len(comm_id_i), len(comm_id_j))
                # if (len(comm_id_i) == 2 or len(comm_id_j) == 2):   # some nodes not assigned to any community
                #     continue
                # comm_id_i = [int(x) for x in comm_id_i.strip('[]').split(', ')]
                # comm_id_j = [int(x) for x in comm_id_j.strip('[]').split(', ')]
                avg_predicted_values = []
                # combinations = list(itertools.product(comm_id_i, comm_id_i))
                # print(len(comm_id_i), len(comm_id_j))
                # for id1, id2 in zip(comm_id_i, comm_id_j):
                # for id1, id2 in combinations:
                representative_node_of_i = cc[cc['Cluster'] == id1][cmeasure].iloc[0]
                representative_node_of_j = cc[cc['Cluster'] == id2][cmeasure].iloc[0]
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
                predicted_value = (statistics.mean([Rici, Rjcj, CiCj]))
                avg_predicted_values.append(predicted_value)
                max_predict = max(avg_predicted_values)
                lst.append([i,j,max_predict])

            predicted_values = pd.DataFrame(lst, columns=cols)
            print('predicted_values', predicted_values)
            predicted_values['TrustValue_new'] = predicted_values['TrustValue'].apply(lambda avg: 1 if avg > 0.60 else 0)
            
            common_pairs = pd.merge(ground_truth, predicted_values, on=['Node1', 'Node2'], how='inner')
            print(common_pairs.shape)
            common_pairs = common_pairs.rename(columns={'TrustValue_x': 'ground_truth', 'TrustValue_new': 'predicted_value','TrustValue_y':'score'})
            filename =  cmeasure + "_predict_ground_truth.csv"
            common_pairs.to_csv(filename)
            
            # Extract ground truth and predicted values for common pairs
            ground_truth_common = common_pairs['ground_truth'].tolist()
            predicted_values_common = common_pairs['predicted_value'].tolist()
          

            # Calculate accuracy for common pairs
            accuracy_common = accuracy_score(ground_truth_common, predicted_values_common)
            print("Centrality Measure", cmeasure)
            print("Accuracy for Common Pairs:", accuracy_common)
    
    
def parse_args():
   parser = argparse.ArgumentParser(description="Read File")
   parser.add_argument("--dataset",type = str)
   parser.add_argument("--graphfile",type = str)
   parser.add_argument("--communityfile",type = str)
   parser.add_argument("--community_center",type = str)
   parser.add_argument("--ratingfile",type = str)
   parser.add_argument("--overlap",type = str)
   return parser.parse_args()


def main():
   inputs=parse_args()
   start_time = time.time()
   prediction(inputs.dataset,inputs.graphfile,inputs.communityfile,inputs.community_center,inputs.ratingfile,inputs.overlap)
   end_time = time.time()
   elapsed_time_seconds = end_time - start_time

    # Convert elapsed time to hours and minutes
   elapsed_hours = int(elapsed_time_seconds // 3600)
   elapsed_minutes = int((elapsed_time_seconds % 3600) // 60)

    # print("Start Time:", start_time)
    # print("End Time:", end_time)
   print("Elapsed Time:", elapsed_hours, "hours", elapsed_minutes, "minutes")
if __name__ == '__main__':
    main()
