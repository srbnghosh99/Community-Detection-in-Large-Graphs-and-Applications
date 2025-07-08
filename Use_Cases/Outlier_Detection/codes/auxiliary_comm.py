import matplotlib.pyplot as plt
import pandas as pd
from networkx.readwrite import json_graph
import json
import networkx as nx
import seaborn as sns
from pathlib import Path
import csv
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
import bson
#import feature_extraction_v2
import feature_extraction
import prediction
import os
import ast



def are_all_same(lst):
    return all(x == lst[0] for x in lst)

def load_solutions2(filename,st,ed):
    """Load list of solutions from a BSON file."""
    with open(filename, "rb") as f:
        data = bson.decode(f.read())
    print(f'Number of total solutions {len(data["data"])}')
    chunk_data = data['data'][st:ed]
    solutions = [item["solution"] for item in chunk_data]
    fitness_values = [item["fitness"] for item in chunk_data]
    # print(len(solutions),len(fitness_values))
    return solutions, fitness_values

def load_solutions(filename):
    """Load list of solutions from a BSON file."""
    with open(filename, "rb") as f:
        data = bson.decode(f.read())
    # print(data)

    solutions = [item["solution"] for item in data["data"]]
    fitness_values = [item["fitness"] for item in data["data"]]
    return solutions, fitness_values

def create_nested_folder(path):
    current_path = ""
    for part in path.split(os.sep):
        if part == "":
            continue  # skip empty parts (for absolute paths)
        current_path = os.path.join(current_path, part)
        if not os.path.exists(current_path):
            os.mkdir(current_path)
            print(f"Created: {current_path}")
        else:
            print(f"Exists: {current_path}")

def create_folder(outdirectory):
    if os.path.exists(outdirectory):
        print('folder exists')
    else:
        try:
            print('create folder')
            os.mkdir(outdirectory)
            print(f"Directory '{outdirectory}' created successfully")
        except FileExistsError:
            print(f"Directory '{outdirectory}' already exists")
        except Exception as e:
            print(f"An error occurred: {e}")

def find_auxiliary_communities_from_bson(graph_file,community_file,output,labelled_file,start,end):
    G = nx.read_edgelist(graph_file, delimiter=' ', nodetype=int)
    print(f'Number of nodes {G.number_of_nodes()} Number of edges {G.number_of_edges()}')
    
    precision_scores_0 = []
    recall_scores_0 = []
    f1_scores_0 = []
    precision_scores_1 = []
    recall_scores_1 = []
    f1_scores_1 = []
    accuracy = []
    

    '''
    solutions, fitness = load_solutions(community_file)
    print(f'Number of solutions: {len(solutions)}, performing now {start} to {end}')
    sorted_pairs = sorted(zip(fitness, solutions), reverse=True, key=lambda x: x[0])

    # Unpack the sorted pairs back into separate lists
    sorted_fitness, sorted_solutions = zip(*sorted_pairs)

    # Convert the tuples back to lists if needed
    sorted_fitness = list(sorted_fitness)
    sorted_solutions = list(sorted_solutions)



    top5perc = round(len(solutions)* (0.05)) 
    bottm5perc = len(sorted_solutions) - top5perc
    #sorted_solutions = sorted_solutions[0:top5perc]
    sorted_solutions = sorted_solutions[bottm5perc:]
    #sorted_solutions = sorted_solutions[start:end]
    #print(len(sorted_solutions))
    '''
    create_nested_folder(output)
    df = pd.read_csv(community_file)
    print(f'Number of solutions: {df.shape[0]}, performing now {start} to {end}')
    df = df[start:end]
    print(df)
    df['Solution'] = df['Solution'].apply(ast.literal_eval)
    sorted_fitness = df['Fitness_val'].tolist()
    df = df.sort_values(by='Fitness_val',ascending=False)
    nodeslis = G.nodes()
    increment = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
    #for solution in tqdm(sorted_solutions, desc="Processing Solutions"):
        #print('solution')
        solution =row['Solution']
        #print(len(nodeslis),len(solution))
        df_comm = pd.DataFrame({'Node':nodeslis,'Community':solution})
        #print(df_comm)
        df_comm = df_comm.sort_values(by=['Node'], key=lambda x: x.astype(int)).reset_index()
        df_comm = df_comm[['Node', 'Community']]
        #print(df_comm)

        count = 0
        auxiliary_communities = []

        node_to_community = df_comm.set_index('Node')['Community'].to_dict()
        nodelis = []
        community_nodes = df_comm.groupby('Community')['Node'].apply(list).to_dict()
        auxiliary_communities = []
        inc = 0
        # Initialize auxiliary communities set
        auxiliary_communities = {}
        auxiliary_community_counter = 1

        #print(f'total_communities {len(community_nodes)}')
        total_communities =len (community_nodes)
        #progress_bar = tqdm(total=total_communities, desc="Processing Nodes", unit="node")
        for community, nodes in community_nodes.items():
            for u in nodes:
                neighbors = list(G.neighbors(u))
                df = df_comm[df_comm['Node'].isin(neighbors)]
                comm = df["Community"].tolist()
                unique_set = set(comm)
                unique_comms = list(unique_set)
                result_different = are_all_same(unique_comms)
                if result_different == False:
                    nodelis.append(u)
                    aux_community_id = f'AUX_{auxiliary_community_counter}'
                    auxiliary_community_counter += 1
                    for v in neighbors:
                        if (node_to_community[u] != node_to_community[v]) and (u != v):
                            auxiliary_communities[aux_community_id] = [u, v]
         #       progress_bar.update()
         #   progress_bar.close()

        combined_communities = {**community_nodes, **auxiliary_communities}
        #print(combined_communities)

        column_names = ['Community', 'Node']
        df = pd.DataFrame(list(combined_communities.items()), columns=column_names)
        df_features = feature_extraction.find_auxiliary_communities(G, df)
        precision_0,recall_0,fscore_0,precision_1,recall_1,fscore_1,acc = prediction.feature_weight_threshold_estimation(df_features, labelled_file)
        precision_scores_0.append(precision_0)
        recall_scores_0.append(recall_0)
        f1_scores_0.append(fscore_0)
        precision_scores_1.append(precision_1)
        recall_scores_1.append(recall_1)
        f1_scores_1.append(fscore_1)
        accuracy.append(acc)
    predicted_Df = pd.DataFrame({
        'fitness_value':sorted_fitness,
        'Accuracy':accuracy,
        'Precision_0': precision_scores_0,  
        'Recall_0': recall_scores_0,
        'F1-score_0': f1_scores_0,
        'Precision_1': precision_scores_1, 
        'Recall_1': recall_scores_1,
        'F1-score_1': f1_scores_1
    })
    print(predicted_Df)
    
    
    #create_folder(output)
    #output = output + '/bottom5perc_' + str(start) +'to'+str(end)+ '_prediction.csv'
    output = output + '/' + str(end) + '_prediction.csv'
    print('output dir',output)
    predicted_Df.to_csv(output)
    





def find_auxiliary_communities(graph_file,community_file,output):

    G = nx.read_edgelist(graph_file,delimiter=' ',nodetype=int)
    print(G.number_of_nodes(), G.number_of_edges())

    df_comm = pd.read_csv(community_file, sep=',')
    #finan_comm = pd.read_csv("/Users/shrabanighosh/Downloads/data/outlier_detection/assignments/finan/louvain_financial_data.csv",sep = ' ')

    # finan_comm.columns = ['Node','Community']
    # finan_comm.sort_values(by=['Community'], key=lambda x: x.astype(int)).reset_index()
    #print(finan_comm)
    df_comm = df_comm.sort_values(by=['Node'], key=lambda x: x.astype(int)).reset_index()
    df_comm = df_comm[['Node','Community']]
    #print(df_comm)

    count = 0
    auxiliary_communities = []

    node_to_community = df_comm.set_index('Node')['Community'].to_dict()
    nodelis = []
    community_nodes = df_comm.groupby('Community')['Node'].apply(list).to_dict()
    auxiliary_communities = []
    inc = 0
    # Initialize auxiliary communities set
    auxiliary_communities = {}
    auxiliary_community_counter = 1


    total_communities = len(community_nodes)
    progress_bar = tqdm(total=total_communities, desc="Processing Nodes", unit="node")
    for community, nodes in community_nodes.items():
        for u in nodes:
            neighbors = list(G.neighbors(u))
            df = df_comm[df_comm['Node'].isin(neighbors)]
            comm = df["Community"].tolist()
            unique_set = set(comm)
            unique_comms = list(unique_set)
            result_different = are_all_same(unique_comms)
            if result_different == False:
                nodelis.append(u)
                aux_community_id = f'AUX_{auxiliary_community_counter}'
                auxiliary_community_counter += 1
                for v in neighbors:
                    if (node_to_community[u] != node_to_community[v]) and (u != v):
                        auxiliary_communities[aux_community_id] = [u,v]
            progress_bar.update()
        progress_bar.close()

    combined_communities = {**community_nodes, **auxiliary_communities}
    #print(combined_communities)

    column_names = ['Community','Node']
    df = pd.DataFrame(list(combined_communities.items()), columns=column_names)

    #print(df)
    df.to_csv(output, index = False)

def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")
    
    parser.add_argument("--graph_file",type = str)
    parser.add_argument("--community_file",type = str)
    parser.add_argument("--flag", type=str)
    parser.add_argument("--outputfile",type = str)
    parser.add_argument("--labelled_file", type=str)
    parser.add_argument("--start_index", type=int)
    parser.add_argument("--end_index", type=int)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.graph_file)
    print(inputs.community_file)
    if inputs.flag == 1:
        find_auxiliary_communities(inputs.graph_file,inputs.community_file,inputs.outputfile)
    else:
        find_auxiliary_communities_from_bson(inputs.graph_file, inputs.community_file, inputs.outputfile,inputs.labelled_file,inputs.start_index,inputs.end_index)

  
if __name__ == '__main__':
    main()
