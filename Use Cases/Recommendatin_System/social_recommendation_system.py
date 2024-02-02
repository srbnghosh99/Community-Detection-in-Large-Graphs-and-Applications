#  <#Title#>

import igraph as ig
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


G = nx.read_edgelist("ciao_trustnet.csv",delimiter=' ', nodetype=int)
print(G.number_of_nodes(), G.number_of_edges(), len(sorted(G.nodes())))

#number of users = 7317
#number of edges = 5205

detected_community_df = pd.read_csv("ciao_trustnet_ego_splitting_membership.csv")

#number of communities = 9056 (higher than number of nodes)

#detected_community_df = pd.read_csv("input")
print("detected_community_df")
print(detected_community_df)

community_mapping = {}
detected_community_df['Community'] = detected_community_df['Community'].apply(ast.literal_eval)
for index, row in detected_community_df.iterrows():
    nodes = row['Node']
    community = row['Community']
    # print(community)
    for c in community:
        # print(c)
        if c in community_mapping:
            community_mapping[c].append(nodes)
        else:
            community_mapping[c] = [nodes]
#community_df = pd.DataFrame(list(community_mapping.items()), columns=['Community', 'Nodes'])
community_df = pd.DataFrame(list(community_mapping.items()), columns=['Community', 'Nodes'])
community_df['count'] = community_df['Nodes'].apply(len)
community_df = community_df.sort_values(by='count', ascending=False)
list_of_communities = community_df['Community'].tolist()
print(community_df)
#print(list_of_communities)
for i in list_of_communities:
#    nodes_to_include = community_df[community_df['Community'] == 1274]['Nodes'].to_list()
    nodes_to_include = community_df.loc[community_df['Community'] == i, 'Nodes'].iloc[0]
#    print(nodes_to_include)
    
        # Create a subgraph from the list of nodes
    subgraph = G.subgraph(nodes_to_include)
#         subgraph.number_of_edges(),subgraph.number_of_nodes()
    json_data = json_graph.node_link_data(subgraph, {'source': 'fromId', 'target': 'toId'})
    outputfile = "community_json/" + "comm_"+ str(i)+ ".json"
    with open(outputfile,'w') as json_file:
        json.dump(json_data,json_file,separators=(',', ':'))
#    break
print("Code executed")

#for i in folder_namelist:
## Example using subprocess to run another Python file
#    file_to_run = "/Users/shrabanighosh/Downloads/ngraph.centrality-main/centrality_measure_copy.py"
#
#    try:
#        subprocess.run(['python3', file_to_run,inputpathfolder,outputpathfolder], check=True)
#    except subprocess.CalledProcessError as e:
#        print(f"Error executing file: {e}")

#print(community_df)

#def propensity of a node():
#    degree_Centrality
#    betweenness degree_Centrality
#    closeness degree_Centrality
#    modularity
#    eigenvector Centrality
#    pagerank
