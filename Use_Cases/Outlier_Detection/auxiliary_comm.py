import igraph as ig
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


def are_all_same(lst):
    return all(x == lst[0] for x in lst)

def find_auxiliary_communities(graph_file,community_file,output):

    G = nx.read_edgelist(graph_file,delimiter=' ',nodetype=int)
    print(G.number_of_nodes(), G.number_of_edges())

    #finan_comm = pd.read_csv("/Users/shrabanighosh/Downloads/data/outlier_detection/assignments/finan/louvain_financial_data.csv",sep = ' ')
    df_comm = pd.read_csv(community_file,sep = ',')
    # finan_comm.columns = ['Node','Community']
    # finan_comm.sort_values(by=['Community'], key=lambda x: x.astype(int)).reset_index()
    #print(finan_comm)
    df_comm = df_comm.sort_values(by=['Node'], key=lambda x: x.astype(int)).reset_index()
    df_comm = df_comm[['Node','Community']]
    print(df_comm)

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
    print(combined_communities)

    column_names = ['Community','Node']
    df = pd.DataFrame(list(combined_communities.items()), columns=column_names)

    print(df)
    df.to_csv(output, index = False)

def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")
    
    parser.add_argument("--graph_file",type = str)
    parser.add_argument("--community_file",type = str)
    parser.add_argument("--outputfile",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.graph_file)
    print(inputs.community_file)
    find_auxiliary_communities(inputs.graph_file,inputs.community_file,inputs.outputfile)
  
if __name__ == '__main__':
    main()
