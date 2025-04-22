import matplotlib.pyplot as plt
import pandas as pd
from networkx.readwrite import json_graph
import json
import networkx as nx
import seaborn as sns
from pathlib import Path
import csv
from collections import Counter
import numpy as np
from surprise import accuracy
from tqdm import tqdm
import statistics
from sklearn.metrics import classification_report
import argparse
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from os.path import dirname, join as pjoin

def calculate_rating_similarity(rating_vector_i, rating_vector_j):
    len_i = len(rating_vector_i)
    len_j = len(rating_vector_j)
    max_len = max(len_i, len_j)
    rating_vector_i = np.pad(rating_vector_i, (0, max_len - len_i))
    rating_vector_j = np.pad(rating_vector_j, (0, max_len - len_j))

    dot_product = np.sum(rating_vector_i * rating_vector_j)
    norm_i = np.sqrt(np.sum(rating_vector_i**2))
    norm_j = np.sqrt(np.sum(rating_vector_j**2))
    if norm_i == 0 or norm_j == 0:
        raise ValueError("Vector norms must be non-zero for similarity calculation.")
    
    rating_similarity = dot_product / (norm_i * norm_j)
    return rating_similarity


def prediction(G,overlap,ground_truth,cd_algo,cc,rating):

    user_ratings = rating.groupby('userid')['rating'].agg(list).reset_index()
    user_ratings['rating_vector'] = user_ratings['rating'].apply(np.array)

    lst = []

    if overlap == 'nonoverlapping':
        with open('report.txt', 'w') as file:
            centrality_measure_list = ['MaxBetweennessNode','MaxSameAsDegreeCentralityNode','MaxOutCentralityNode','MaxinCentralityNode','RandomNode']
            for cmeasure in centrality_measure_list:
                file.write(cmeasure + '\n')
                for index, row in tqdm(ground_truth.iterrows(), total=len(ground_truth)):
                    i = row['Node1']
                    j = row['Node2']
                    id1 = cd_algo[cd_algo['Node'] == i]['Community'].iloc[0]
                    id2 = cd_algo[cd_algo['Node'] == j]['Community'].iloc[0]
                    avg_predicted_values = []
                    representative_node_of_i = cc[cc['Cluster'] == id1][cmeasure].iloc[0]
                    representative_node_of_j = cc[cc['Cluster'] == id2][cmeasure].iloc[0]
                    user_vector = user_ratings[user_ratings['userid'] == i]['rating_vector'].iloc[0]
                    center_vector_i = user_ratings[user_ratings['userid'] == representative_node_of_i]['rating_vector'].iloc[0]
                    
                    Rici = calculate_rating_similarity(user_vector,center_vector_i)
                    user_vector = user_ratings[user_ratings['userid'] == j]['rating_vector'].iloc[0]
                    center_vector_j = user_ratings[user_ratings['userid'] == representative_node_of_j]['rating_vector'].iloc[0]
                    Rjcj = calculate_rating_similarity(user_vector,center_vector_j)
                    CiCj = calculate_rating_similarity(center_vector_i,center_vector_j)
                    valuelist = [Rici,Rjcj,CiCj]
                    predicted_value = (statistics.mean([Rici, Rjcj, CiCj]))
                    avg_predicted_values.append(predicted_value)
                    max_predict = max(avg_predicted_values)
                    lst.append([i,j,max_predict])
                cols=['Node1', 'Node2', 'TrustValue']
                predicted_values = pd.DataFrame(lst, columns=cols)
                predicted_values['TrustValue_new'] = predicted_values['TrustValue'].apply(lambda avg: 1 if avg > 0.55 else 0)
                common_pairs = pd.merge(ground_truth, predicted_values, on=['Node1', 'Node2'], how='inner')
                common_pairs = common_pairs.rename(columns={'TrustValue_x': 'ground_truth', 'TrustValue_new': 'predicted_value','TrustValue_y':'score'})
                filename =  cmeasure + "_predict_ground_truth.csv"
                ground_truth_common = common_pairs['ground_truth'].tolist()
                predicted_values_common = common_pairs['predicted_value'].tolist()
                report = classification_report(ground_truth_common, predicted_values_common, labels=[0,1])
                print(report)
                
                file.write(report)

    if overlap == 'overlapping':
        with open('report.txt', 'w') as file:
            centrality_measure_list = ['MaxBetweennessNode','MaxSameAsDegreeCentralityNode','MaxOutCentralityNode','MaxinCentralityNode','RandomNode']
            for cmeasure in centrality_measure_list:
                file.write(cmeasure + '\n')
                for index, row in tqdm(ground_truth.iterrows(), total=len(ground_truth)):
                    i = row['Node1']
                    j = row['Node2']
                    if i not in cd_algo['Node'].values or j not in cd_algo['Node'].values:
                        print('Does not exist')
                        continue
                    comm_id_i = cd_algo[cd_algo['Node'] == i]['Community'].iloc[0]
                    comm_id_j = cd_algo[cd_algo['Node'] == j]['Community'].iloc[0]
                    if (len(comm_id_i) == 2 or len(comm_id_j) == 2):   # some nodes not assigned to any community
                        continue
                    comm_id_i = [int(x) for x in comm_id_i.strip('[]').split(', ')]
                    comm_id_j = [int(x) for x in comm_id_j.strip('[]').split(', ')]
                    avg_predicted_values = []
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
                filename =   cmeasure + "_predict_ground_truth.csv"
                common_pairs.to_csv(filename)

                # Extract ground truth and predicted values for common pairs
                ground_truth_common = common_pairs['ground_truth'].tolist()
                predicted_values_common = common_pairs['predicted_value'].tolist()
                report = classification_report(ground_truth_common, predicted_values_common, labels=[0,1])
                print(report)
                file.write(report)


def parse_args():
   parser = argparse.ArgumentParser(description="Read File")
   parser.add_argument("--dataset",type = str)
   parser.add_argument("--graphfile",type = str)
   parser.add_argument("--communityfile",type = str)
   parser.add_argument("--community_center",type = str)
   parser.add_argument("--ratingfile",type = str)
   parser.add_argument("--overlap",type = str)
   parser.add_argument("--groundtruthfile",type = str)
   return parser.parse_args()


def main():
   inputs=parse_args()
   curr_directory = os.getcwd()
   graphfile = pjoin(curr_directory,inputs.dataset,inputs.graphfile)
   communityfile = pjoin(curr_directory,inputs.dataset, inputs.communityfile)
   community_center = pjoin(curr_directory,inputs.dataset, inputs.community_center)
   ratingfile = pjoin(curr_directory,inputs.dataset, inputs.ratingfile)
   groundtruthfile = pjoin(curr_directory,inputs.dataset, inputs.groundtruthfile)
   ground_truth = pd.read_csv(groundtruthfile)
   G = nx.read_edgelist(graphfile,delimiter=',', nodetype=int)
   cd_algo = pd.read_csv(communityfile)
   cc = pd.read_csv(community_center) 
   rating = pd.read_csv(ratingfile)
   prediction(G,inputs.overlap,ground_truth,cd_algo,cc,rating)

if __name__ == '__main__':
    main()

