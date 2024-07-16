import networkx as nx
import pandas as pd
import ast
from tqdm import tqdm

G = nx.read_weighted_edgelist("/Users/shrabanighosh/Downloads/data/outlier_detection/Financial_dat_assignments/DGraph_fin_nxjaccard_weighted.edgelist",nodetype=int)
print(G.number_of_nodes(), G.number_of_edges())

result_dict = {}
count = 0
with open('Financial_dat_assignments/DGraph_fin_clique_percolation_k3_v2.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('[')
        node = parts[0].strip()
        # node = int(node)
        communities_str = '[' + parts[1]
        node = int(node)
        # Use ast.literal_eval to parse the list safely
        communities = ast.literal_eval(communities_str)
        # print(node, communities)

        if not G.has_node(node):
            count = count + 1
            # print(f"Node {node} from the file is not present in the graph.")
        # print(communities)
        result_dict[node] = communities
        # break

# Sort the dictionary based on nodes
sorted_result_dict = dict(sorted(result_dict.items()))

# Extract community lists from the dictionary
community_lists = [communities for communities in result_dict.values()]


# Flatten the list of community lists
flat_communities = [community for sublist in community_lists for community in sublist]

# Get unique community values
unique_communities = set(flat_communities)

# print("Unique Communities:", (unique_communities))

# Create a mapping from old community values to new community values starting from 1
community_mapping = {old_comm: new_comm for new_comm, old_comm in enumerate(unique_communities, start=1)}

# Adjust communities in the result_dict
node_dict = {node: [community_mapping[old_comm] for old_comm in communities] for node, communities in result_dict.items()}

# Print the adjusted result_dict
# for node, communities in adjusted_result_dict.items():
#     print(f"{node}: {communities}")

# Extract community lists from the dictionary
community_lists = [communities for communities in node_dict.values()]

# Flatten the list of community lists
flat_communities = [community for sublist in community_lists for community in sublist]

# Get unique community values
unique_communities = set(flat_communities)

print("Unique Communities:", len(unique_communities))

# Convert the set to a sorted list
sorted_communities = sorted(unique_communities)

# Print the first 100 items
# print(sorted_communities[:100])

# Sort the dictionary based on nodes
sorted_node_dict = dict(sorted(node_dict.items()))
# Print the sorted dictionary
# for node, communities in sorted_adjusted_result_dict.items():
#     print(f"{node}: {communities}")
#     break

community_dict = {}

for node, communities in node_dict.items():
    for community in communities:
        if community not in community_dict:
            community_dict[community] = []
        community_dict[community].append(node)

def value_for_f1(communities):
    community_length = len(communities)
    return community_length

def value_for_f2(node,communities):
    # print(node,communities)
    num_unique_communities = len(communities)
    num_neighbors = list(G.neighbors(node))
    # print(num_unique_communities,num_neighbors)
    if len(num_neighbors) != 0:
        ratio = num_unique_communities / len(num_neighbors)
    return ratio


def value_for_f3(node_to_calculate):
    clustering_coefficient = nx.clustering(G, node_to_calculate)
    return 1-clustering_coefficient

def value_for_f4(node_to_calculate):
    degree_of_node = G.degree(node_to_calculate)
    # Calculate the sum of edge weights incident to the chosen node
    sum_of_edge_weights = sum(data['weight'] for _, _, data in G.edges(data=True) if node_to_calculate in [_, _])
    # print(sum_of_edge_weights)
    # Calculate the degree of the chosen node
    # Calculate Deg(V)/Sum of Edge Weights
    if sum_of_edge_weights != 0:
        ratio = degree_of_node / sum_of_edge_weights
    else:
        ratio = 0
    return ratio
    
def value_for_f5(node,communities):
    total_score = 0

    for community in communities:
        deg_v = G.degree(node)
        n = len(community_dict[community])

        # Calculate the normalized degree score
        score = deg_v / (n * (n - 1) / 2)

        total_score += score

    # Calculate the aggregate score (you can customize this based on your requirements)
    aggregate_score = total_score

    return aggregate_score



def value_for_f6(node,community_nodes):
    total_score = 0

    for community in communities:
        deg_v = G.degree(node)
        n = len(community_dict[community])

        # Calculate the normalized degree score
        score = deg_v / (n * (n - 1))
        total_score += score
    aggregate_score = total_score

    return aggregate_score

# Printing the created nested dictionary
# Creating a nested dictionary for nodes and their features
graph_features = {}

# Example functions for feature values (replace these with your actual functions)


count = 0
length_of_dict = len(node_dict.items())
progress_bar = tqdm(total=length_of_dict, desc="Processing Nodes", unit="node")
# Example feature values for each node
for node, communities in node_dict.items():
    feature_values = {
        'f1': value_for_f1(communities),
        'f2': value_for_f2(node,communities),
        'f3': value_for_f3(node),
        'f4': value_for_f4(node),
        'f5': value_for_f5(node,communities),
        'f6': value_for_f6(node,communities)
    }
    graph_features[node] = feature_values
    progress_bar.update()
progress_bar.close()
print("Done")
print(graph_features)


print(f" Total {count} number of nodes does not exist in original graph")

import concurrent.futures
from tqdm import tqdm

import concurrent.futures
from tqdm import tqdm

def calculate_features(node, communities):
    return node, {
        'f1': value_for_f1(communities),
        'f2': value_for_f2(node, communities),
        'f3': value_for_f3(node),
        'f4': value_for_f4(node),
        'f5': value_for_f5(node, communities),
        'f6': value_for_f6(node, communities)
    }

# Assuming node_dict is your original dictionary
with concurrent.futures.ProcessPoolExecutor() as executor, tqdm(total=len(node_dict)) as progress_bar:
    # Use executor.starmap to parallelize the computation of feature values
    results = list(executor.starmap(calculate_features, node_dict.items()))
    progress_bar.update(len(node_dict))

# Convert the results back to a dictionary
graph_features = dict(results)
