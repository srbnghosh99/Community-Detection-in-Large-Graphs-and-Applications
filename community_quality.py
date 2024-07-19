import networkx as nx
import json
from networkx.algorithms.community.quality import modularity
import os
from networkx.algorithms.cluster import average_clustering
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
import pandas as pd


def community_comparison(graphfile,directory_path):

    # Load graph from txt file
    # graph = nx.read_edgelist("/Users/shrabanighosh/Downloads/data/recommendation_system/ciao/renumbered_graph_ciao.csv", nodetype=int)
    # directory_path = "/Users/shrabanighosh/Downloads/data/recommendation_system/ciao/louvain"

    graph = nx.read_edgelist(graphfile, nodetype=int)


    

    # Initialize an empty list to store communities
    # communities = []

    # Iterate through each file in the directory
    # for filename in os.listdir(directory_path):
    #     if filename.endswith(".json"):
    #         file_path = os.path.join(directory_path, filename)
    #         with open(file_path, "r") as f:
    #             community_data = json.load(f)
    #             communities.append(community_data)  # Append each community data to the list

    communities = {}

    no_comm = len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]) - 1

    print(f"Number of communities in '{directory_path}': {no_comm}")

    # Iterate through each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as f:
                community_data = json.load(f)
                community_nodes = {node["id"] for node in community_data["nodes"]}
                communities[filename] = community_nodes

    # Print each community
    # for idx, (name, community) in enumerate(communities.items(), start=1):
    #     print(f"Community {idx} ({name}):")
    #     # print(community)
    #     print()  

    # Combine all communities into a list of sets
    communities_list = list(communities.values())

    # Calculate modularity
    modularity_value = modularity(graph, communities_list)

    print(f"Modularity: {modularity_value}")

    cluster_coeff_lis = []
    density_lis = []
    conductance_lis = []
    coverage_lis = []
    edge_cut_lis = []
    normalized_cut_lis = []



    # Calculate and print the average clustering coefficient for each community
    for idx, (name, community) in enumerate(communities.items(), start=1):
        subgraph = graph.subgraph(community)
        avg_clustering_coefficient = average_clustering(subgraph)
        print(f"Average clustering coefficient for Community {idx} ({name}): {avg_clustering_coefficient}")
        cluster_coeff_lis.append(avg_clustering_coefficient)

    # Optionally, calculate the average clustering coefficient for the whole graph
    overall_avg_clustering_coefficient = average_clustering(graph)
    print(f"Overall average clustering coefficient for the entire graph: {overall_avg_clustering_coefficient}")


    def compute_internal_density(graph, community_nodes):
        subgraph = graph.subgraph(community_nodes)
        num_edges = subgraph.number_of_edges()
        num_nodes = subgraph.number_of_nodes()
        if num_nodes > 1:
            internal_density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
        else:
            internal_density = 0  # Internal density is 0 if there's only one node
        return internal_density

    # Calculate and print the internal density for each community
    for idx, (name, community) in enumerate(communities.items(), start=1):
        internal_density = compute_internal_density(graph, community)
        print(f"Internal density for Community {idx} ({name}): {internal_density}")
        density_lis.append(internal_density)

    # Optionally, calculate the overall internal density for the whole graph
    overall_internal_density = compute_internal_density(graph, graph.nodes())
    print(f"Overall internal density for the entire graph: {overall_internal_density}")


    # Function to compute the conductance of a community
    def compute_conductance(graph, community_nodes):
        cut_size = nx.cut_size(graph, community_nodes)
        vol_S = sum(d for n, d in graph.degree(community_nodes))
        vol_not_S = sum(d for n, d in graph.degree() if n not in community_nodes)
        conductance_value = cut_size / min(vol_S, vol_not_S)
        return conductance_value

    # Calculate and print the conductance for each community
    for idx, (name, community) in enumerate(communities.items(), start=1):
        conductance_value = compute_conductance(graph, community)
        print(f"Conductance for Community {idx} ({name}): {conductance_value}")
        conductance_lis.append(conductance_value)

    # Optionally, calculate the overall conductance for the whole graph
    # overall_conductance_value = compute_conductance(graph, graph.nodes())
    # print(f"Overall conductance for the entire graph: {overall_conductance_value}")

    # Function to compute the coverage of a community
    def compute_coverage(graph, community_nodes):
        subgraph = graph.subgraph(community_nodes)
        num_edges_within = subgraph.number_of_edges()
        total_edges = graph.number_of_edges()
        coverage_value = num_edges_within / total_edges
        return coverage_value

    # Calculate and print the coverage for each community
    for idx, (name, community) in enumerate(communities.items(), start=1):
        coverage_value = compute_coverage(graph, community)
        print(f"Coverage for Community {idx} ({name}): {coverage_value}")
        coverage_lis.append(coverage_value)

    # Optionally, calculate the overall coverage for the whole graph
    overall_coverage_value = compute_coverage(graph, graph.nodes())
    print(f"Overall coverage for the entire graph: {overall_coverage_value}")


    # Function to compute the edge cut of a community
    def compute_edge_cut(graph, community_nodes):
        return nx.cut_size(graph, community_nodes)

    # Function to compute the normalized cut of a community
    def compute_normalized_cut(graph, community_nodes):
        cut_size = nx.cut_size(graph, community_nodes)
        vol_S = sum(d for n, d in graph.degree(community_nodes))
        vol_not_S = sum(d for n, d in graph.degree() if n not in community_nodes)
        normalized_cut_value = (cut_size / vol_S) + (cut_size / vol_not_S)
        return normalized_cut_value

    # Calculate and print the edge cut and normalized cut for each community
    for idx, (name, community) in enumerate(communities.items(), start=1):
        edge_cut_value = compute_edge_cut(graph, community)
        normalized_cut_value = compute_normalized_cut(graph, community)
        print(f"Edge Cut for Community {idx} ({name}): {edge_cut_value}")
        print(f"Normalized Cut for Community {idx} ({name}): {normalized_cut_value}")
        edge_cut_lis.append(edge_cut_value)
        normalized_cut_lis.append(normalized_cut_value)

# Optionally, calculate the overall edge cut and normalized cut for the whole graph
# overall_edge_cut_value = compute_edge_cut(graph, graph.nodes())
# overall_normalized_cut_value = compute_normalized_cut(graph, graph.nodes())
# print(f"Overall Edge Cut for the entire graph: {overall_edge_cut_value}")
# print(f"Overall Normalized Cut for the entire graph: {overall_normalized_cut_value}")

    # Initialize an empty DataFrame
    # Names for each row

    data = {
    'Number of communities': no_comm,
    'Modularity': modularity_value,
    'cluster_coeff_lis': cluster_coeff_lis,
    'overall_avg_clustering_coefficient': overall_avg_clustering_coefficient,
    'density_lis': density_lis,
    'overall_internal_density': overall_coverage_value,
    'conductance_lis': conductance_lis,
    'coverage_lis': coverage_lis,
    'overall_coverage_value': overall_coverage_value,
    'edge_cut_lis': edge_cut_lis,
    'normalized_cut_lis': normalized_cut_lis
    }
    
    # data = {
    #     'Number of communities', 'Modularity', 'cluster_coeff_lis', 'overall_avg_clustering_coefficient', 'density_lis',
    #              'overall_internal_density', 'conductance_lis','coverage_lis',  'overall_coverage_value', 'edge_cut_lis',
    #                'normalized_cut_lis'
    # }

    # Convert all single values to lists
    data = {key: value if isinstance(value, list) else [value] for key, value in data.items()}

    # Find the maximum length of lists
    max_length = max(len(value) for value in data.values())

    # Pad lists with None to make them of equal length
    padded_data = {key: value + [None] * (max_length - len(value)) for key, value in data.items()}

    # Create DataFrame from the dictionary
    df = pd.DataFrame(padded_data)

    # Print the DataFrame
    print(df)
    filename = directory_path + '/community_metrics.csv'
    df.to_csv(filename)



def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--graphfile",type = str)
    parser.add_argument("--directory_path",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    community_comparison(inputs.graphfile,inputs.directory_path)
  

if __name__ == '__main__':
    main()
