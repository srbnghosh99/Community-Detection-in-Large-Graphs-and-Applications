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

    #number of users = 7317
    #number of edges = 5205

    detected_community_df = pd.read_csv(cd_file)

    #number of communities = 9056 (higher than number of nodes)

    #detected_community_df = pd.read_csv("input")
    print("detected_community_df")
    print(detected_community_df)

    community_mapping = {}

    if (overlap == 'overlapping'):
        detected_community_df['Community'] = detected_community_df['Community'].apply(ast.literal_eval)
        print(detected_community_df)
        
      
        for index, row in detected_community_df.iterrows():
            nodes = row['Node']
            community = row['Community']
            print(community)
            for c in community:
                print(c)
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
            outputfile = outdirectory + "comm_"+ str(i)+ ".json"
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
            outputfile = outdirectory +  "comm_"+ str(i)+ ".json"
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
    parser.add_argument("--outdir",type = str)
    parser.add_argument("--overlap",type = str)
    return parser.parse_args()

def main():

    start_time = time.time()
    inputs=parse_args()
    print(inputs.graphfile)
    print(inputs.cdfile)
    print(inputs.outdir)
    create_community_propensity(inputs.dataset,inputs.graphfile,inputs.cdfile,inputs.outdir,inputs.overlap)
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



