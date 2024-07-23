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
import os
import time
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



def community_vis(graphfile,inputdir,outdir,overlapping):
    curr_directory = os.getcwd()
    directory = pjoin(curr_directory,inputdir)
    graphfile = pjoin(curr_directory,graphfile)
    outdir = pjoin(curr_directory,outdir)
    create_folder(outdir)
    clear_folder(outdir)

    dfs = []
    # Lists to store metrics
    
    nodes_count_dict = {}
    degree_dict = {}
    density_dict = {}
    centralization_dict = {}
    clustering_coefficients_dict = {}

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):  # Assuming the files are CSV files
            file_path = os.path.join(directory, filename)
            print(file_path)
            detected_community_df = pd.read_csv(file_path)
            print(detected_community_df)
            comm = []
            cc = []
            cen = []
            deg = []
            den = []
            nodes_count = []
            
            G = nx.read_edgelist(graphfile,delimiter=' ', nodetype=int)
            
            df = pd.read_csv("email-Eu-core.txt",sep = ' ')
            if (overlapping == "False"):
                print("Non-Overlapping")
                
        #       detected_community_df = pd.read_csv("community_clusters/gemsec-email-Eu-core.csv")
#                detected_community_df = pd.read_csv(file_path,sep = ' ')
                
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
                    print("nodes_to_extract",len(nodes_to_extract))
                    subgraph = G.subgraph(nodes_to_extract)
                    print("nodes",subgraph.number_of_nodes(),subgraph.number_of_edges())
                    subgraph_nodes = nodes_to_extract
                    
                    print("subgraph_nodes",len(subgraph_nodes))
                    print("Cluster",community )
#                    if len(subgraph_nodes) > 0:
                    avgclus = nx.average_clustering(subgraph)
                    print('Clustering coefficient: ', avgclus)
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
                    comm.append(community)
                    cc.append(avgclus)
                    cen.append(centralization)
                    deg.append(average_neighbor_degree)
                    nodes_count.append(count)
                    den.append(density)
                
                filename = os.path.splitext(filename)[0]
                centralization_dict[filename] = [cen]
                clustering_coefficients_dict[filename] = [cc]
                nodes_count_dict[filename] = [nodes_count]
                degree_dict[filename] = [deg]
                density_dict[filename] = [den]
                print(centralization_dict)
                df_centralization = pd.DataFrame({k: pd.Series(v[0]) for k, v in centralization_dict.items()})
                df_clustering_coefficients = pd.DataFrame({k: pd.Series(v[0]) for k, v in clustering_coefficients_dict.items()})
                df_degree = pd.DataFrame({k: pd.Series(v[0]) for k, v in degree_dict.items()})
                df_density = pd.DataFrame({k: pd.Series(v[0]) for k, v in density_dict.items()})
                df_nodes = pd.DataFrame({k: pd.Series(v[0]) for k, v in nodes_count_dict.items()})


                print(df_clustering_coefficients,df_centralization)
                df_clustering_coefficients.to_csv(outdir+'/Clustering_coefficients.csv', index = False)
                df_centralization.to_csv(outdir+'/Centralization.csv', index = False)
                df_degree.to_csv(outdir+'/Degree.csv', index = False)
                df_density.to_csv(outdir+'/Density.csv', index = False)
                df_nodes.to_csv(outdir+'/Nodes_count_each_community.csv', index = False)

            else:
                print("Overlapping")
                detected_community_df = pd.read_csv(file_path)
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
                    count = row['count']
                    community = row['Community']
        #            nodes_to_extract = detected_community_df[detected_community_df['Community'] == community]['Nodes']
                    subgraph = G.subgraph(nodes_to_extract)
                    subgraph_nodes = nodes_to_extract
                    print("Cluster", community)
                    print("==============")
                    avgclus = nx.average_clustering(subgraph)
                    print('cluster coefficient',avgclus)
                    degree_centrality = nx.degree_centrality(subgraph)
                    centralization = sum([(max(degree_centrality.values()) - degree_centrality[node]) for node in subgraph_nodes]) / ((len(subgraph_nodes) - 1) * (len(subgraph_nodes) - 2))
                    print('centralization', centralization)
                    avg_neighbor_degrees = nx.average_neighbor_degree(subgraph)
                    average_neighbor_degree = sum(avg_neighbor_degrees.values()) / len(avg_neighbor_degrees)
                    print('Avg neighbors degree', average_neighbor_degree)
                    density = 2 * subgraph.number_of_edges() / (subgraph.number_of_nodes() * (subgraph.number_of_nodes() - 1))
                    print('Density', density)
                    comm.append(community)
                    cc.append(avgclus)
                    cen.append(centralization)
                    deg.append(average_neighbor_degree)
                    nodes_count.append(count)
                    den.append(density)
#            nx.draw_networkx(subgraph)
#            plt.show()
                filename = os.path.splitext(filename)[0]
                centralization_dict[filename] = [cen]
                clustering_coefficients_dict[filename] = [cc]
                nodes_count_dict[filename] = [nodes_count]
                degree_dict[filename] = [deg]
                density_dict[filename] = [den]
                print(centralization_dict)
                df_centralization = pd.DataFrame({k: pd.Series(v[0]) for k, v in centralization_dict.items()})
                df_clustering_coefficients = pd.DataFrame({k: pd.Series(v[0]) for k, v in clustering_coefficients_dict.items()})
                df_degree = pd.DataFrame({k: pd.Series(v[0]) for k, v in degree_dict.items()})
                df_density = pd.DataFrame({k: pd.Series(v[0]) for k, v in density_dict.items()})
                df_nodes = pd.DataFrame({k: pd.Series(v[0]) for k, v in nodes_count_dict.items()})


                print(df_clustering_coefficients,df_centralization)
                df_clustering_coefficients.to_csv(outdir+'Clustering_coefficients.csv', index = False)
                df_centralization.to_csv(outdir +'/Centralization.csv', index = False)
                df_degree.to_csv(outdir +'/Avg_neighbor_degree.csv', index = False)
                df_density.to_csv(outdir +'/Density.csv', index = False)
                df_nodes.to_csv(outdir +'/Nodes_count_each_community.csv', index = False)
        
def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--graphfile",type = str)
    parser.add_argument("--inputdir",type = str)
    parser.add_argument("--outdir",type = str)
    parser.add_argument("--overlapping",type = str)

    return parser.parse_args()

def main():
    inputs=parse_args()
#    print(inputs.inputfilename)
#    print(inputs.overlapping)
    community_vis(inputs.graphfile,inputs.inputdir,inputs.outdir,inputs.overlapping)
  

if __name__ == '__main__':
    main()
