from communities.algorithms import spectral_clustering
import numpy as np
from communities.algorithms import louvain_method
from community import community_louvain
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



def eig_laplacian(A, k=2):
    n = np.shape(A)[0]
    D = np.diag(1 / np.sqrt(np.ravel(A.sum(axis=0))))
    L = np.identity(n) - D.dot(A).dot(D)
    return eigsh(L, k, which='SM')
    
def spectral_clust(A, k=5):
    n = np.shape(A)[0]
    V, Z = eig_laplacian(A, k)
    
    rows_norm = np.linalg.norm(Z, axis=1, ord=2)
    Y = (Z.T / rows_norm).T
    centroids, distortion = kmeans(Y, k)
    
    y_hat = np.zeros(n, dtype=int)
    for i in range(n):
        dists = np.array([np.linalg.norm(Y[i] - centroids[c]) for c in range(k)])
        y_hat[i] = np.argmin(dists)
    # print(y_hat)
    return y_hat

def spectral_clust_chauduri(A, tau, k=5):
    n = np.shape(A)[0]
    At = A + tau / n
    D = np.diag(1 / np.sqrt(np.ravel(At.sum(axis=0))))
    L = np.identity(n) - D.dot(A).dot(D)
    V, Z = eigsh(L, k, which='SM')
    
    rows_norm = np.linalg.norm(Z, axis=1, ord=2)
    Y = (Z.T / rows_norm).T
    centroids, distortion = kmeans(Y, k)
    
    y_hat = np.zeros(n, dtype=int)
    for i in range(n):
        dists = np.array([np.linalg.norm(Y[i] - centroids[c]) for c in range(k)])
        y_hat[i] = np.argmin(dists)
    # print(y_hat)
    return y_hat

def read_graph(filepath, source_graph=None):
    # Use the largest connected component from the graph
    if source_graph is None:
        graph = nx.read_edgelist(filepath, nodetype=int).to_undirected()
    else:
        graph = source_graph
    # graph = max(nx.connected_components(graph), key=len)
    Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
    G0 = graph.subgraph(Gcc[0])
    print(G0.number_of_nodes())
    
    # Split the graph edges into train and test
    # random_edges = list(graph.edges())
    random_edges = list(G0.edges())
    np.random.shuffle(random_edges)
    train_edges = random_edges[:graph.number_of_edges()//2]
    test_edges = random_edges[graph.number_of_edges()//2:]
    # print(train_edges,test_edges)
    
    # Create the training graph
    train_graph = nx.Graph()
    train_graph.add_edges_from(train_edges)
    # train_graph = max(nx.connected_components(train_graph), key=len)
    Gcc = sorted(nx.connected_components(train_graph), key=len, reverse=True)
    G0 = graph.subgraph(Gcc[0])
    train_graph = G0
    
    # Create the test graph
    test_graph = nx.Graph()
    # test_graph.add_nodes_from(train_graph.nodes())
    test_graph.add_nodes_from(list(train_graph))
    test_graph.add_edges_from(test_edges)
    
    return train_graph, test_graph


def spec_algos(inputf,outputf):
    G = nx.read_edgelist(inputf,nodetype=int, data=(("Weight",int),))
    # adj_matrix = [...]
    adj_matrix = nx.to_numpy_array(G)
    S = nx.to_numpy_array(G)

    van_labels = spectral_clustering(S,5)
    # print('van_labels',van_labels)

    taos = np.exp(np.linspace(-7, 12, 400))
    reg_sizes = []
    conducts = []
    conducts_test = []

    N = G.number_of_nodes()
    tao_heurisitc = G.number_of_edges() * 2 / N
    nodes = np.array(G.nodes())
    accuracies = []
    accuracies_ch = []
    for tao in taos:
        SR = S + tao / N 
        y_hat = spectral_clust(SR)
        y_hat_ch = spectral_clust_chauduri(S, tao)
    print((y_hat))
    print((y_hat_ch))

    communities = spectral_clustering(adj_matrix, k=5)
    # print(communities)
    df = pd.DataFrame()
    lis1 = []
    lis2 = []
    print("Non - overlapping Clustering")
    k = 0
    nodelis = []
    commlist = []
    for i in communities:
        # print(i)
        l = [k] * len(i)
        k = k+1
        nodelis.append(i)
        commlist.append(l)

    flatlist1=[element for sublist in nodelis for element in sublist]
    flatlist2=[element for sublist in commlist for element in sublist]

    df = pd.DataFrame(
        {'authoname': flatlist1,
        'community': flatlist2,
        })
    # print(flatlist1,flatlist2)
    # df.to_csv("rudi_spectral.csv")
    df.to_csv(outputf)


def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")

    parser.add_argument("--inputfile",type = str)
    parser.add_argument("--outputfile",type = str)
    
    return parser.parse_args()

def main():
    inputs=parse_args()
    print("Input file name:: ", inputs.inputfile)

    print("Output file name:: ",inputs.outputfile)
    spec_algos(inputs.inputfile,inputs.outputfile)
    train_graph, test_graph = read_graph(inputs.inputfile)
  

if __name__ == '__main__':
    main()

