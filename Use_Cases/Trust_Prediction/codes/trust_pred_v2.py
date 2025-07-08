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
from tqdm import tqdm
import statistics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

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


def prediction(ground_truth, cd_algo,cc,rating):
    #print(ground_truth)
    #print(cc)
    #print(cd_algo)
    #print(rating)
    df1 = ground_truth[ground_truth['TrustValue'] == 1][:6700]
    df2 = ground_truth[ground_truth['TrustValue'] == 0][:3300]
    ground_truth = pd.concat([df2,df1])
    rating['userid'] = rating['userid'].astype(int)
    user_ratings = rating.groupby('userid')['rating'].agg(list).reset_index()
    user_ratings['rating_vector'] = user_ratings['rating'].apply(np.array)
    cc['Cluster'] = cc['Cluster'].astype(int)

    lst = []
    centrality_measure_list = ['MaxClosenessNode', 'MaxSameAsDegreeCentralityNode', 'MaxBetweennessNode',
                               'MaxOutCentralityNode', 'MaxinCentralityNode', 'RandomNode']
    f1_scores_0 = []
    precision_scores_0 = []
    recall_scores_0 = []
    auc_scores = []
    precision_scores_1 = []
    recall_scores_1 = []
    f1_scores_1 = []
    cmeasure_name = []
    #for cmeasure in tqdm(centrality_measure_list, total=len(centrality_measure_list)):
    for cmeasure in centrality_measure_list:
        for index, row in ground_truth.iterrows():
            
            i = row['Node1']
            j = row['Node2']
            #print(i,j)
            id1 = cd_algo[cd_algo['Node'] == i]['Community'].iloc[0]
            id2 = cd_algo[cd_algo['Node'] == j]['Community'].iloc[0]
            #print(id1,id2)
            #id1 = str(id1)
            #id2 = str(id2) 
            avg_predicted_values = []
            representative_node_of_i = cc[cc['Cluster'] == id1][cmeasure].iloc[0]
            representative_node_of_j = cc[cc['Cluster'] == id2][cmeasure].iloc[0]
            #print(representative_node_of_i,representative_node_of_j)
            representative_node_of_i = int(representative_node_of_i)
            representative_node_of_j = int(representative_node_of_j)
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
        #report = classification_report(ground_truth_common, predicted_values_common, labels=[0,1])
        #print(report)
        report = classification_report(ground_truth_common, predicted_values_common, labels=[0,1], target_names=['Class 0', 'Class 1'],output_dict=True)
        auc = roc_auc_score(ground_truth_common, predicted_values_common)
        precision_scores_0.append(report['Class 0']['precision'])
        recall_scores_0.append(report['Class 0']['recall'])
        f1_scores_0.append(report['Class 0']['f1-score'])
        precision_scores_1.append(report['Class 1']['precision'])
        recall_scores_1.append(report['Class 1']['recall'])
        f1_scores_1.append(report['Class 1']['f1-score'])
        auc_scores.append(auc)
        cmeasure_name.append(cmeasure)
    df = pd.DataFrame({
        'Precision_Class_0': precision_scores_0,
        'Recall_Class_0': recall_scores_0,
        'F1_Class_0': f1_scores_0,
        'Precision_Class_1': precision_scores_1,
        'Recall_Class_1': recall_scores_1,
        'F1_Class_1': f1_scores_1,
        'AUC': auc_scores,
        'Measure': cmeasure_name
    })
    
    return df

'''
cc = pd.read_csv('ciao/subgraphs_propensity/centerclusters.csv')
rating = pd.read_csv('ciao/rating.csv')
ground_truth = pd.read_csv('ciao/ciao_groudtruth.csv')
cd_algo = pd.read_csv('random_node_comm.csv',sep = ',')
prediction(ground_truth,cd_algo,cc,rating)
'''
