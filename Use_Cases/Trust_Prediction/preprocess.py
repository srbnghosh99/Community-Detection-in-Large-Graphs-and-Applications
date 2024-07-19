#  <#Title#>

import pandas as pd
import networkx as nx
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import networkx as nx
import os
from os.path import dirname, join as pjoin
import scipy.io as sio
from collections import Counter
import matplotlib.pyplot as plt


def raw_file_read(directory,dataset):

    directory = os.getcwd()
    # mat_fname = "/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/rating.mat"
    mat_fname = pjoin(directory,dataset, 'rating.mat')
    mat_contents = sio.loadmat(mat_fname)
    rating_array = mat_contents['rating']
    rating_df = pd.DataFrame(rating_array, columns=['userid', 'productid', 'categoryid', 'rating', 'helpfulness'])
    csv_fname = mat_fname.replace(".mat", ".csv")
    rating_df.to_csv(csv_fname,index = False)
    # mat_fname = "/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/trustnetwork.mat"
    mat_fname = pjoin(directory,dataset, 'trustnetwork.mat')
    mat_contents = sio.loadmat(mat_fname)
    trustnetwork_array = mat_contents['trustnetwork']
    trustnet_df = pd.DataFrame(trustnetwork_array)
    csv_fname = mat_fname.replace(".mat", ".csv")
    

    ### Renumber nodes
    G = nx.read_edgelist(csv_fname,delimiter=' ', nodetype=int)
    print("Number of nodes",G.number_of_nodes(), "Number of edges",G.number_of_edges())
    nodes_with_less_than_two_edges = [node for node, degree in G.degree() if degree < 2]
    print('number of users with less than two trustors',len(nodes_with_less_than_two_edges))
    G.remove_nodes_from(nodes_with_less_than_two_edges)
    df_edges = nx.to_pandas_edgelist(G)
    remaining_nodes = sorted(list(G.nodes()))
    mapping = {node: new_number + 1 for new_number, node in enumerate(remaining_nodes)}
    G_renumbered = nx.relabel_nodes(G, mapping)
    # Write the renumbered graph to a file
    output_file = "renumbered_graph_" + dataset + ".csv"
    filename = pjoin(directory,dataset, output_file)
    #distribution of trustor and trustee
    column_names = ['Trustor', 'Trustee']
    trustnet_df = pd.read_csv(filename,names=column_names,delimiter=' ')
    
    nx.write_edgelist(G_renumbered, filename, delimiter=' ', data=False)
    print("Done")


    #rating dataset
    product_rating = rating_df['productid'].value_counts().reset_index()
    processed_ciao_rating = product_rating[product_rating['count'] > 1]
    output_file = "processed_" + dataset + "_rating.csv"
    rating_filename = pjoin(directory,dataset, output_file)
    rating_df[rating_df['productid'].isin(processed_ciao_rating['count'].tolist())].to_csv(rating_filename,index = False)



    
    
    freq = trustnet_df['Trustor'].value_counts().reset_index()
    graph = G_renumbered
    freq = trustnet_df['Trustor'].value_counts().reset_index()

    print('Clustering coefficient: ', nx.average_clustering(graph))
    density = 2 * graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1))
    degrees = dict(G.degree())
    # Calculate the average degree
    average_degree = sum(degrees.values()) / len(degrees)

    print("Degrees of vertices:", degrees)
    print("Average degree:", average_degree)
    data_distribution(directory,dataset,filename)


def data_distribution(directory,dataset,filename):
    epi = nx.read_edgelist(filename,delimiter=' ',create_using=nx.DiGraph, nodetype=int)
    print("Number of nodes",epi.number_of_nodes(), "Number of edges",epi.number_of_edges())
    trustor_counts = Counter(dict(epi.in_degree()).values())
    trustee_counts = Counter(dict(epi.out_degree()).values())
    sorted_by_keys = sorted(trustor_counts.items())
    total_sum = sum(trustor_counts.values())
    print("Number of Trustor", total_sum)

    sorted_by_keys = sorted(trustee_counts.items())
    total_sum = sum(trustee_counts.values())
    print("Number of Trustee", total_sum)

    # Convert trustor and trustee counts to lists for plotting
    epi_trustor_degrees, epi_trustor_counts = zip(*sorted(trustor_counts.items()))
    epi_trustee_degrees, epi_trustee_counts = zip(*sorted(trustee_counts.items()))
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].plot(epi_trustor_degrees, epi_trustor_counts, label='Trustors', color='b', marker='o', linestyle='')
    axs[0].set_xlabel('Degree of Trustors', fontsize=14)
    axs[0].set_ylabel('Count of Trustors', fontsize=14)
    axs[0].set_title(dataset.upper() +' Trustor Distribution', fontsize=16)
    axs[0].legend()
    axs[0].grid(True)

    # Plot the trustee distribution
    axs[1].plot(epi_trustee_degrees, epi_trustee_counts, label='Trustees', color='r', marker='o', linestyle='')
    axs[1].set_xlabel('Degree of Trustees', fontsize=14)
    axs[1].set_ylabel('Count of Trustees', fontsize=14)
    axs[1].set_title(dataset.upper() +' Trustee Distribution', fontsize=16)
    axs[1].legend()
    axs[1].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    plot_filename = pjoin(directory,dataset, 'data_distribution.png')
    plt.savefig(plot_filename)
    plt.show()

    

def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--dataset",type = str)
    parser.add_argument("--directory",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    raw_file_read(inputs.directory,inputs.dataset)



if __name__ == '__main__':
    main()
