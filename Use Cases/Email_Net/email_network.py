# Title

import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
from networkx.readwrite import json_graph
import json
import networkx as nx
import seaborn as sns
from pathlib import Path
import csv
import matplotlib.pyplot as plt  # Optional, for plotting
import ast
import argparse
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def community_vis(input,overlapping):
    print(overlapping)
    G = nx.read_edgelist("email-Eu-core.txt",delimiter=' ', nodetype=int)
    #df = pd.read_csv("email-Eu-core.txt",sep = ' ')
    if (overlapping == "False"):
        print("Non Overlapping")
        
#       detected_community_df = pd.read_csv("community_clusters/gemsec-email-Eu-core.csv")
        detected_community_df = pd.read_csv(input,sep = ',')
        print("Number of communities ", detected_community_df["Community"].nunique())
        print(detected_community_df)
#        result = detected_community_df['Community'].value_counts().reset_index()
#        filtered_result = result[result['Community'] > 10]
        filtered_result= detected_community_df['Community'].value_counts().reset_index().query('count > 10').reset_index(drop=True)
        print(filtered_result)
        for index, row in filtered_result.iterrows():
            community = row['Community']
            count = row['count']
            nodes_to_extract = detected_community_df[detected_community_df['Community'] == community]['Node'].tolist()
            subgraph = G.subgraph(nodes_to_extract)
            subgraph_nodes = nodes_to_extract
            print("Cluster",community )
            
            print('Clustering coefficient: ', nx.average_clustering(subgraph))
            degree_centrality = nx.degree_centrality(subgraph)
            centralization = sum([(max(degree_centrality.values()) - degree_centrality[node]) for node in subgraph_nodes]) / ((len(subgraph_nodes) - 1) * (len(subgraph_nodes) - 2))
            print('centralization', centralization)
            avg_neighbor_degrees = nx.average_neighbor_degree(subgraph)
            average_neighbor_degree = sum(avg_neighbor_degrees.values()) / len(avg_neighbor_degrees)
            print('Avg neighbors degree', average_neighbor_degree)
            density = 2 * subgraph.number_of_edges() / (subgraph.number_of_nodes() * (subgraph.number_of_nodes() - 1))
            print('Nodes',count)
            print('Density', density)
            print("==============")
#            nx.draw_networkx(subgraph)
#            plt.show()

    else:
        print(" Overlapping")
        detected_community_df = pd.read_csv(input)
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
        community_df = pd.DataFrame(list(community_mapping.items()), columns=['Community', 'Nodes'])
        community_df = pd.DataFrame(list(community_mapping.items()), columns=['Community', 'Nodes'])
        community_df['count'] = community_df['Nodes'].apply(len)
        community_df = community_df.sort_values(by='count', ascending=False)
        print(community_df)
        filtered_result = community_df.query('count > 10').reset_index(drop=True)
        # Display the community DataFrame
        print(filtered_result)
        for index, row in filtered_result.iterrows():
            nodes_to_extract = row['Nodes']
            community = row['Community']
#            nodes_to_extract = detected_community_df[detected_community_df['Community'] == community]['Nodes']
            subgraph = G.subgraph(nodes_to_extract)
            subgraph_nodes = nodes_to_extract
            print("Cluster", community)
            print("==============")
            print('cluster coefficient',nx.average_clustering(subgraph))
            degree_centrality = nx.degree_centrality(subgraph)
            centralization = sum([(max(degree_centrality.values()) - degree_centrality[node]) for node in subgraph_nodes]) / ((len(subgraph_nodes) - 1) * (len(subgraph_nodes) - 2))
            print('centralization', centralization)
            avg_neighbor_degrees = nx.average_neighbor_degree(subgraph)
            average_neighbor_degree = sum(avg_neighbor_degrees.values()) / len(avg_neighbor_degrees)
            print('Avg neighbors degree', average_neighbor_degree)
            density = 2 * subgraph.number_of_edges() / (subgraph.number_of_nodes() * (subgraph.number_of_nodes() - 1))
            print('Density', density)
#            nx.draw_networkx(subgraph)
#            plt.show()
        
def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--inputfilename",type = str)
    parser.add_argument("--overlapping",type = str)
#    parser.add_argument("--outputfilename",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.inputfilename)
    print(inputs.overlapping)
    community_vis(inputs.inputfilename,inputs.overlapping)
  

if __name__ == '__main__':
    main()
