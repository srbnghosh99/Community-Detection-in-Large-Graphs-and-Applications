#  <#Title#>

import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
from networkx.readwrite import json_graph
import json
import networkx as nx
import seaborn as sns
from pathlib import Path


#lp = pd.read_csv('/Users/shrabanighosh/Downloads/data/outlier detection/assignments/label_propagation_DGraph_fin_header_false_100.csv')
#
#lv = pd.read_csv('/Users/shrabanighosh/Downloads/data/outlier detection/assignments/louvain_financial_data.csv',sep =' ')
#
##print(lp['Node'].value_counts())
##print(lv['Node'].value_counts())
#
#print(len(lp[lp['Community'] == 1]['Node'].tolist()))

'''
class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_edge(self, u, v):
        if u not in self.adj_list:
            self.adj_list[u] = []
        self.adj_list[u].append(v)

    def has_self_loop(self):
        self_loops = 0
        for vertex in self.adj_list:
            if vertex in self.adj_list[vertex]:
                self_loops += 1
#                return True
        return self_loops

def read_graph_from_file(file_path):
    graph = Graph()
    with open(file_path, 'r') as file:
        for line in file:
            vertices = list(map(int, line.strip().split()))
            vertex = vertices[0]
            for neighbor in vertices[1:]:
                graph.add_edge(vertex, neighbor)
    return graph

## Example usage
#if __name__ == "__main__":
#    file_path = '/Users/shrabanighosh/Downloads/LabelPropagation-master/data/DGraph_fin_header_false.edgelist'  # Replace with your file path
#    graph = read_graph_from_file(file_path)
#
#    # Check for self-loops
##    if graph.has_self_loop():
##        print("The graph has self-loops.")
##    else:
##        print("The graph does not have self-loops.")
#
#    # Count self-loops
#    self_loops_count = graph.has_self_loop()
#    print(f"The graph has {self_loops_count} self-loops.")




def count_self_loops(graph):
    return graph.number_of_selfloops()

# Example usage
if __name__ == "__main__":
    file_path = '/Users/shrabanighosh/Downloads/LabelPropagation-master/data/release-flickr-links.edgelist'  # Replace with your file path
    graph = read_graph_from_file(file_path)

    # Convert the custom graph to a NetworkX graph
    nx_graph = nx.Graph(graph.adj_list)

    # Count self-loops using NetworkX function
    self_loops_count = count_self_loops(nx_graph)
    print(f"The graph has {self_loops_count} self-loops.")


import networkx as nx

class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_edge(self, u, v):
        if u not in self.adj_list:
            self.adj_list[u] = []
        self.adj_list[u].append(v)

def read_graph_from_file(file_path):
    graph = Graph()
    with open(file_path, 'r') as file:
        for line in file:
            vertices = list(map(int, line.strip().split()))
            vertex = vertices[0]
            for neighbor in vertices[1:]:
                graph.add_edge(vertex, neighbor)
    return graph

# Example usage
if __name__ == "__main__":
    file_path = '/Users/shrabanighosh/Downloads/LabelPropagation-master/data/DGraph_fin_header_false.edgelist'  # Replace with your file path
#    custom_graph = read_graph_from_file(file_path)
    G = nx.read_edgelist(file_path, delimiter=' ', nodetype=int)
    # G = nx.read_weighted_edgelist(file_path)
    print(list(nx.selfloop_edges(G)))
    # Count self-loops using NetworkX function
#    self_loops_count = G.number_of_selfloops()
#    print(f"The graph has {self_loops_count} self-loops.")

'''


# Assuming you have a graph in some form, you can create a NetworkX graph
# Replace this with your graph loading logic or use any existing graph
# G = nx.read_edgelist('your_edgelist.txt', nodetype=int)

# def calculate_jaccard_similarity(graph):
#     jaccard_similarity = nx.jaccard_coefficient(graph)
#     for u, v, coef in jaccard_similarity:
#         print(f'Jaccard similarity between nodes {u} and {v} {coef}')

file_path = '/Users/shrabanighosh/Downloads/LabelPropagation-master/data/release-flickr-links.edgelist'  # Replace with your file path
#    custom_graph = read_graph_from_file(file_path)
G = nx.read_edgelist(file_path, delimiter=' ', nodetype=int)


def calculate_jaccard_similarity_for_existing_edges(graph, output_file):
    with open(output_file, 'w') as file:
        for u, v in graph.edges():
            if graph.has_edge(u, v):
                jaccard_similarity = nx.jaccard_coefficient(graph, [(u, v)])
                for _, _, coef in jaccard_similarity:
                    file.write(f'{u} {v} {coef}\n')

output_file_path = '/Users/shrabanighosh/Downloads/LabelPropagation-master/data/release-flickr-links_nxjaccard_weighted.edgelist'
# Calculate Jaccard similarity only for existing edges
calculate_jaccard_similarity_for_existing_edges(G, output_file_path)
print("Complete program")
