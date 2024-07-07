
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.cluster.vq import whiten, kmeans
from numpy import linalg as LA
from networkx.algorithms.cuts import conductance
from scipy.sparse.linalg import eigsh
import communities
from communities.algorithms import spectral_clustering
import time



def spectral_algo(inputfile,outputfile):
    # Read the edge list and construct the graph
    G = nx.read_edgelist(inputfile, delimiter=' ', nodetype=int)
    print(G.number_of_nodes(),G.number_of_edges())
    # Extract the nodes as features and reshape to 2D array
    nodes = list(G.nodes())
    nodes = np.array(nodes).reshape(-1, 1)
    nodelist = list(G.nodes())
    # Construct nearest neighbors graph
    n_neighbors = 10
    affinity_matrix = kneighbors_graph(nodes, n_neighbors, mode='connectivity', include_self=True)
    
    # Convert the sparse matrix to a dense array
    affinity_array = affinity_matrix.toarray()
    
    # Spectral clustering with a fixed number of clusters
    n_clusters = 10  # You can adjust this value based on your problem domain or use a heuristic method
    labels = spectral_clustering(affinity_array, n_clusters=n_clusters, eigen_solver='arpack', random_state=42)
    
    # Visualize the graph and color nodes based on the spectral clustering result
    #pos = nx.spring_layout(G)
    #nx.draw(G, pos, node_color=labels, cmap=plt.cm.Blues, with_labels=True)
    #plt.show()
    print(len(labels), len(nodes))
    df = pd.DataFrame({'Node': nodelist, 'Cluster': labels})
    print(df)
    df.to_csv(outputfile, index = False)


def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")

    parser.add_argument("--inputfilename",type = str)
    parser.add_argument("--outputfilename",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.inputfilename)
    print(inputs.outputfilename)
    spectral_algo(inputs.inputfilename,inputs.outputfilename)
  

if __name__ == '__main__':
    main()

