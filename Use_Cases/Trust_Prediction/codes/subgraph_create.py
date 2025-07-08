import matplotlib.pyplot as plt
import pandas as pd
from networkx.readwrite import json_graph
import json
import networkx as nx
import seaborn as sns
from pathlib import Path
import csv
import ast
import subprocess
import argparse
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime, timedelta
import time
import os
from os.path import dirname, join as pjoin
import bson
import create_node_propensity
import find_center_of_communities
import trust_pred_v2
from tqdm import tqdm

def clear_folder(outdirectory):
    print('Clear Folder')
    # Check if the folder exists
    if os.path.exists(outdirectory):
        # Remove all files in the folder
        for filename in os.listdir(outdirectory):
            file_path = os.path.join(outdirectory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"The folder {outdirectory} does not exist.")
        

def create_folder(outdirectory):
    print('create folder')
    try:
        os.mkdir(outdirectory)
        print(f"Directory '{outdirectory}' created successfully")
    except FileExistsError:
        print(f"Directory '{outdirectory}' already exists")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_solutions_v1(filename):
    """Load list of solutions from a BSON file."""
    with open(filename, "rb") as f:
        data = bson.decode(f.read())
    return data["solutions"]

def load_solutions(filename):
    """Load list of solutions from a BSON file."""
    with open(filename, "rb") as f:
        data = bson.decode(f.read())
    # print(data)

    solutions = [item["solution"] for item in data["data"]]
    fitness_values = [item["fitness"] for item in data["data"]]
    return solutions, fitness_values

def create_community_propensity_from_bson(dataset,graph_file, cd_file, outdirectory,ratingfile,ground_truthfile,outputdir,start,end):

    directory = os.getcwd()
    # mat_fname = "/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/rating.mat"
    graph_file = pjoin(directory, dataset, graph_file)
    cd_file = pjoin(directory,dataset, cd_file)
    ratingfile = pjoin(directory, dataset, ratingfile)
    ground_truthfile = pjoin(directory, dataset, ground_truthfile)
    #outdirectory = pjoin(directory, dataset, outdirectory)
    #outdirectory = pjoin(outdirectory)
    outdirectory = outdirectory + '_' + str(start) + 'to' + str(end)
    create_folder(outdirectory)
    clear_folder(outdirectory)
    newoutdir =  outdirectory + '_propensity' + '_' + str(start) + 'to' + str(end) 
    create_folder(newoutdir)
    G = nx.read_edgelist(graph_file, delimiter=' ', nodetype=int)
    print(f'Number of nodes {G.number_of_nodes()} Number of edges {G.number_of_edges()}')
    ground_truth = pd.read_csv(ground_truthfile)
    rating = pd.read_csv(ratingfile)
    df = pd.read_csv(cd_file)
    #print(f'Number of solutions: {df.shape[0]}, performing now {start} to {end}')
    df = df[start:end]
    df['Solution'] = df['Solution'].apply(ast.literal_eval)
    sorted_fitness = df['Fitness_val'].tolist()
    df = df.sort_values(by='Fitness_val',ascending=False)
    df_master = pd.DataFrame()
    '''
    # solutions, _ = load_solutions(cd_file)
    solutions = load_solutions_v1(cd_file)
    print(f'Number of solutions: {len(solutions)}')
    for solution in solutions:
    '''
    nodes = G.nodes()
    for index, row in tqdm(df.iterrows(), total=len(df)):
        solution =row['Solution']
        #print(len(solution),len(nodes))
        if len(set(solution)) == len(nodes):
            continue

        df_comm = pd.DataFrame({'Node':nodes,'Community':solution})
        df_comm = df_comm.sort_values(by=['Node'], key=lambda x: x.astype(int)).reset_index()
        df_comm = df_comm[['Node', 'Community']]
        nodelis = []
        # community_df = df_comm.groupby('Community')['Node'].apply(list).to_dict()
        community_df = df_comm.groupby('Community')['Node'].apply(list).reset_index()
        #print(community_df) 
        community_df['count'] = community_df['Node'].apply(len)
        community_df = community_df.sort_values(by='count', ascending=False)
        list_of_communities = community_df['Community'].tolist()
        for i in list_of_communities:
            nodes_to_include = community_df.loc[community_df['Community'] == i, 'Node'].iloc[0]
            subgraph = G.subgraph(nodes_to_include)
            json_data = json_graph.node_link_data(subgraph, {'source': 'fromId', 'target': 'toId'})
            outputfile = outdirectory + "/comm_" + str(i) + ".json"
            with open(outputfile, 'w') as json_file:
                json.dump(json_data, json_file, separators=(',', ':'))  
        create_node_propensity.node_propensity(dataset,outdirectory,newoutdir)
        community_center = find_center_of_communities.center_cluster_calculate(dataset,newoutdir)
        community_center = community_center.reset_index()
        df_new = trust_pred_v2.prediction(ground_truth,df_comm,community_center,rating)
        clear_folder(newoutdir)
        clear_folder(outdirectory)
        df_master = pd.concat([df_master, df_new], ignore_index=True)
    create_folder(outputdir)
    filename = outputdir +'/' + str(start) +'to'+str(end)+ '_prediction.csv'
    df_master.to_csv(filename)
    #filename =  str(start) +'to'+str(end)+ '_prediction.csv'
    #df_master.to_csv(filename)

def create_community_propensity(dataset, graph_file, cd_file, outdirectory, overlap):

    directory = os.getcwd()
    # mat_fname = "/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/rating.mat"
    graph_file = pjoin(directory,dataset, graph_file)
    cd_file = pjoin(directory,dataset, cd_file)
    outdirectory = pjoin(directory,dataset, outdirectory)

    
    create_folder(outdirectory)
    clear_folder(outdirectory)
    start_time = datetime.now()
    G = nx.read_edgelist(graph_file,delimiter=' ', nodetype=int)
    print(G.number_of_nodes(), G.number_of_edges(), len(sorted(G.nodes())))


    detected_community_df = pd.read_csv(cd_file,sep = ' ')
    print("detected_community_df")
    print(detected_community_df)

    community_mapping = {}

    if (overlap == 'overlapping'):
        detected_community_df['Community'] = detected_community_df['Community'].apply(ast.literal_eval)
        print(detected_community_df)
        
      
        for index, row in detected_community_df.iterrows():
            nodes = row['Node']
            community = row['Community']
#            print(community)
            for c in community:
#                print(c)
                if c in community_mapping:
                    community_mapping[c].append(nodes)
                else:
                    community_mapping[c] = [nodes]
        #community_df = pd.DataFrame(list(community_mapping.items()), columns=['Community', 'Nodes'])
        community_df = pd.DataFrame(list(community_mapping.items()), columns=['Community', 'Nodes'])
        community_df['count'] = community_df['Nodes'].apply(len)
        community_df = community_df.sort_values(by='count', ascending=False)
        
        
#        community_df = detected_community_df
        list_of_communities = community_df['Community'].tolist()
        print(community_df)
        print(list_of_communities)
        for i in list_of_communities:
        #    nodes_to_include = community_df[community_df['Community'] == 1274]['Nodes'].to_list()
            nodes_to_include = community_df.loc[community_df['Community'] == i, 'Nodes'].iloc[0]
        #    print(nodes_to_include)
            
                # Create a subgraph from the list of nodes
            subgraph = G.subgraph(nodes_to_include)
        #         subgraph.number_of_edges(),subgraph.number_of_nodes()
            json_data = json_graph.node_link_data(subgraph, {'source': 'fromId', 'target': 'toId'})
            outputfile = outdirectory + "/comm_"+ str(i)+ ".json"
            with open(outputfile,'w') as json_file:
                json.dump(json_data,json_file,separators=(',', ':'))
        #    break
        print('no_of_communities',list_of_communities)
        print("Code executed")
    else:
        # Group by 'Community' and aggregate 'Node' into lists
        community_df = detected_community_df.groupby('Community')['Node'].apply(list).reset_index()
        print(community_df)
        # detected_community_df['Community']
        # for index, row in detected_community_df.iterrows():
        #     nodes = row['Node']
        #     community = row['Community']
        #     # print(community)
        #     for c in community:
        #         # print(c)
        #         if c in community_mapping:
        #             community_mapping[c].append(nodes)
        #         else:
        #             community_mapping[c] = [nodes]
        #community_df = pd.DataFrame(list(community_mapping.items()), columns=['Community', 'Nodes'])
        # community_df = pd.DataFrame(list(community_mapping.items()), columns=['Community', 'Nodes'])
        community_df['count'] = community_df['Node'].apply(len)
        community_df = community_df.sort_values(by='count', ascending=False)
        list_of_communities = community_df['Community'].tolist()
        print(community_df)
        for i in list_of_communities:
            nodes_to_include = community_df.loc[community_df['Community'] == i, 'Node'].iloc[0]
            subgraph = G.subgraph(nodes_to_include)
        #         subgraph.number_of_edges(),subgraph.number_of_nodes()
            json_data = json_graph.node_link_data(subgraph, {'source': 'fromId', 'target': 'toId'})
            outputfile = outdirectory +  "/comm_"+ str(i)+ ".json"
            with open(outputfile,'w') as json_file:
                json.dump(json_data,json_file,separators=(',', ':'))
        #    break
        print("Code executed")
        end_time = datetime.now()


def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--dataset",type = str)
    parser.add_argument("--graphfile",type = str)
    parser.add_argument("--cdfile",type = str)
    parser.add_argument("--outdirectory",type = str)
    parser.add_argument("--overlap",type = str,required=False)
    parser.add_argument("--flag", type=str)
    parser.add_argument("--ratingfile",type = str)
    parser.add_argument("--ground_truthfile",type = str)
    parser.add_argument("--outputdir",type = str)
    parser.add_argument("--start_index", type=int)
    parser.add_argument("--end_index", type=int)

    return parser.parse_args()

def main():

    start_time = time.time()
    inputs=parse_args()
#    print(inputs.graphfile)
#    print(inputs.cdfile)
#    print(inputs.outdir)
    if inputs.flag == 1:
        create_community_propensity(inputs.dataset,inputs.graphfile,inputs.cdfile,inputs.outdirectory,inputs.overlap,inputs.ratingfile,inputs.ground_truthfile)
    else:
        create_community_propensity_from_bson(inputs.dataset,inputs.graphfile, inputs.cdfile, inputs.outdirectory,inputs.ratingfile,inputs.ground_truthfile,inputs.outputdir,inputs.start_index,inputs.end_index)

    # Get the end time
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



