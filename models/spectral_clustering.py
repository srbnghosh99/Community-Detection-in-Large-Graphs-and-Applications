#  <#Title#>

import time
import os
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt

from scipy.cluster.vq import whiten, kmeans
from numpy import linalg as LA
from networkx.algorithms.cuts import conductance
from scipy.sparse.linalg import eigsh

%matplotlib inline
matplotlib.rcParams.update({'font.size': 20})

def eig_laplacian(A, k=2):
    n = np.shape(A)[0]
    D = np.diag(1 / np.sqrt(np.ravel(A.sum(axis=0))))
    L = np.identity(n) - D.dot(A).dot(D)
    return eigsh(L, k, which='SM')
    
def spectral_clust(A, k=2):
    n = np.shape(A)[0]
    V, Z = eig_laplacian(A, k)
    
    rows_norm = np.linalg.norm(Z, axis=1, ord=2)
    Y = (Z.T / rows_norm).T
    centroids, distortion = kmeans(Y, k)
    
    y_hat = np.zeros(n, dtype=int)
    for i in range(n):
        dists = np.array([np.linalg.norm(Y[i] - centroids[c]) for c in range(k)])
        y_hat[i] = np.argmin(dists)
    return y_hat

def spectral_clust_chauduri(A, tau, k=2):
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
    return y_hat
    
# Load the data
def read_graph(filepath, source_graph=None):
    # Use the largest connected component from the graph
    if source_graph is None:
        graph = nx.read_edgelist(filepath, nodetype=int).to_undirected()
    else:
        graph = source_graph
    # graph = max(nx.connected_components(graph), key=len)
    Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
    G0 = graph.subgraph(Gcc[0])
    # G0.number_of_nodes()
    
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
    test_graph.add_nodes_from(train_graph.nodes())
    # test_graph.add_nodes_from(list(train_graph))
    test_graph.add_edges_from(test_edges)
    
    return train_graph, test_graph

def generate_dangling_random_graph(partitions=2):
    graph = nx.gaussian_random_partition_graph(partitions * 100, 100, 10, 0.4, 0.05)
    nx.add_path(graph, [1, 300, 1234, 500, 280, 267, 221, 267, 221])
    nx.add_path(graph, [2, 301, 1231, 501, 281, 261, 222, 261, 222])
    nx.add_path(graph, [3, 3012, 12312, 5012, 2812, 2612, 2222, 5021, 3021])
    return graph


def get_min_part_size(labels):
    return min(np.sum(labels), np.size(labels) - np.sum(labels))

def get_avg_min_part_size(map_entry):
    van_part, reg_part = [], []
    for seed in range(0, 1):
        np.random.seed(seed)
        train, _ = read_graph(graph_map[map_entry])
        # print("Part 1")
        N = train.number_of_nodes()
        # S = nx.to_numpy_matrix(train)
        S = nx.to_numpy_array(train)
        tao = train.number_of_edges() * 2 / N
        SR = S + tao / N
        
        van_labels = spectral_clust(S)
        reg_labels = spectral_clust(SR)
        
        van_part.append(get_min_part_size(van_labels))
        reg_part.append(get_min_part_size(reg_labels))
    
    return np.mean(van_part), np.mean(reg_part), N

def graphs_part_sizes():
    van_sizes, reg_sizes, graph_sizes = [], [], []
    for graph_key in graph_map:
        print('Processing ' + graph_map[graph_key])
        van, reg, size = get_avg_min_part_size(graph_key)
        van_sizes.append(van)
        reg_sizes.append(reg)
        graph_sizes.append(size)
    return van_sizes, reg_sizes, graph_sizes
    

van_sizes, reg_sizes, sizes = graphs_part_sizes()


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
maxim = max(np.max(van_sizes), np.max(reg_sizes))
ax.plot([0, np.max(reg_sizes)], [0, np.max(reg_sizes)], color='black')
ax.set_xlabel('Regularised')
ax.set_ylabel('Vanilla')
ax.set_title('Size of the smallest partition')
ax.scatter(reg_sizes, van_sizes, sizes, alpha=0.5)

def get_conductance(map_entry):
    van_cond_train, van_cond_test = [], []
    reg_cond_train, reg_cond_test = [], []
    for seed in range(0, 1):
        np.random.seed(seed)
        train, test = read_graph(graph_map[map_entry])
        
        N = train.number_of_nodes()
        S = nx.to_numpy_array(train)
        tao = train.number_of_edges() * 2 / N
        SR = S + tao / N
        
        van_labels = spectral_clust(S)
        reg_labels = spectral_clust(SR)
        
        nodes = np.array(list(train.nodes()))
        van_cond_train.append(conductance(train, nodes[van_labels == 1]))
        van_cond_test.append(conductance(test, nodes[van_labels == 1]))
        reg_cond_train.append(conductance(train, nodes[reg_labels == 1]))
        reg_cond_test.append(conductance(test, nodes[reg_labels == 1]))
    
    return np.mean(van_cond_train), np.mean(reg_cond_train), np.mean(van_cond_test), np.mean(reg_cond_test), N
    
def graphs_conductances():
    van_train, reg_train, van_test, reg_test, sizes = [], [], [], [], []
    for graph_key in graph_map:
        print('Processing graph ' + graph_map[graph_key])
        vt, rt, vtes, rtes, s = get_conductance(graph_key)
        van_train.append(vt)
        reg_train.append(rt)
        van_test.append(vtes)
        reg_test.append(rtes)
        sizes.append(s)
        
    return van_train, reg_train, van_test, reg_test, sizes

van_train, reg_train, van_test, reg_test, sizes = graphs_conductances()

fig, ax = plt.subplots(1, 1, figsize=(10, 10));
minim = min(np.min(van_train), np.min(reg_train))
maxim = max(np.max(van_train), np.max(reg_train))
ax.plot([minim, maxim], [minim, maxim], color='black')
ax.grid(True)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Regularised')
ax.set_ylabel('Vanilla')
ax.set_title('Conductance on the training graph')
ax.scatter(reg_train, van_train, sizes, alpha=0.5)

fig, ax = plt.subplots(1, 1, figsize=(10, 10));
minim = min(np.min(van_test), np.min(reg_test))
maxim = max(np.max(van_test), np.max(reg_test))
ax.plot([minim, maxim], [minim, maxim], color='black')
ax.grid(True)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Regularised')
ax.set_ylabel('Vanilla')
ax.set_title('Conductance on the testing graph')
ax.scatter(reg_test, van_test, np.array(sizes) * 0.5, alpha=0.5, )



van_part, reg_part = [], []

for seed in range(0, 1):
    np.random.seed(seed)
    train, _ = read_graph(graph_map[graph_key])
    
    N = train.number_of_nodes()
    S = nx.to_numpy_array(train)
    tao = train.number_of_edges() * 2 / N
    SR = S + tao / N
    
    van_labels = spectral_clust(S)
    reg_labels = spectral_clust(SR)
    
    van_part.append(get_min_part_size(van_labels))
    reg_part.append(get_min_part_size(reg_labels))

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
maxim = max(np.max(van_sizes), np.max(reg_sizes))
ax.plot([0, np.max(reg_sizes)], [0, np.max(reg_sizes)], color='black')
ax.set_xlabel('Regularised')
ax.set_ylabel('Vanilla')
ax.set_title('Size of the smallest partition')
ax.scatter(reg_sizes, van_sizes, sizes, alpha=0.5)

fig, ax = plt.subplots(1, 1, figsize=(10, 10));
minim = min(np.min(van_test), np.min(reg_test))
maxim = max(np.max(van_test), np.max(reg_test))
ax.plot([minim, maxim], [minim, maxim], color='black')
ax.grid(True)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Regularised')
ax.set_ylabel('Vanilla')
ax.set_title('Conductance on the testing graph')
ax.scatter(reg_test, van_test, np.array(sizes) * 0.5, alpha=0.5, )


def benchmark(fct, runs=5):
    times = []
    for run in range(runs):
        start = time.time()
        fct()
        end = time.time()
        times.append(end-start)
    return np.mean(times)

def graphs_benchmark():
    vant, regt, sizes = [], [], []
    for graph_key in graph_map:
        print('Processing graph ' + graph_map[graph_key])
       
        train, _ = read_graph(graph_map[graph_key])
        N = train.number_of_nodes()
        S = nx.to_numpy_array(train)
        tao = 2 * train.number_of_edges() / N
        SR = S + tao / N

        vant.append(benchmark(lambda: eig_laplacian(S)))
        regt.append(benchmark(lambda: eig_laplacian(SR)))
        sizes.append(train.number_of_nodes())

    return vant, regt, sizes
    
    
van_times, reg_time, sizes = graphs_benchmark()

fig, ax = plt.subplots(1, 1, figsize=(10, 10));
maxim = max(np.max(van_times), np.max(reg_time))
ax.plot([0.01, maxim], [0.01, maxim], color='black')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True)
ax.set_xlabel('Regularised')
ax.set_ylabel('Vanilla')
ax.set_title('Eigendecomposition processing time')
ax.scatter(reg_time, van_times, sizes, alpha=0.5)

#Analyse an individual graph in detail

np.random.seed(8)
# train_graph, test_graph = read_graph(graph_map[4])
# train_graph, test_graph = read_graph('', nx.karate_club_graph())
# train_graph, test_graph = read_graph('', nx.davis_southern_women_graph())
# train_graph, test_graph = read_graph('', nx.florentine_families_graph())
train_graph, test_graph = read_graph('', generate_dangling_random_graph())
# train_graph, test_graph = read_graph('', nx.read_gml('./dataset/polblogs.gml').to_undirected())


# Compare the eigenvectors of the markov matrix of the vanilla and regularised graphs
S = nx.to_numpy_array(train_graph)

# Tao is initialised with the average degree of the graph
N = train_graph.number_of_nodes()
tao = 2.25
SR = S + tao / N

val_van, vec_van = eig_laplacian(S, 100)
val_reg, vec_reg = eig_laplacian(SR, 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3));
ax1.bar(np.arange(len(vec_van[:, 1])), vec_van[:, 1])
ax1.set_xlabel('Second eigenvector values')
ax1.set_ylabel('Magnitude')
ax2.bar(np.arange(len(vec_reg[:, 1])), vec_reg[:, 1])
ax2.set_xlabel('Second eigenvector values')
ax2.set_ylabel('Magnitude')

# Compute the number of nodes in the smallest partition set.
van_labels = spectral_clust(S)
reg_labels = spectral_clust(SR)

# Compute the size of the minimum partition
van_min_size = min(np.sum(van_labels), np.sum(1 - van_labels))
reg_min_size = min(np.sum(reg_labels), np.sum(1 - reg_labels))
print("Vanilla smallest partition size: {}".format(van_min_size))
print("CoreCut smallest partition size: {}".format(reg_min_size))


# Compute the conductange of the smallest set
nodes = np.array(list(train_graph.nodes()))
van_nodes_with_label = nodes[van_labels == 1]
reg_nodes_with_label = nodes[reg_labels == 1]

print("Vanilla conductance on train: {}".format(conductance(train_graph, van_nodes_with_label)))
print("CoreCut conductance on train: {}".format(conductance(train_graph, reg_nodes_with_label)))
print("Vanilla conductance on test: {}".format(conductance(test_graph, van_nodes_with_label)))
print("CoreCut conductance on test: {}".format(conductance(test_graph, reg_nodes_with_label)))

def node_colors(labels):
    colors = np.empty_like(labels, dtype=object)
    colors[labels == 1] = 'orange'
    colors[labels == 0] = 'cornflowerblue'
    return colors

def draw_bipartite_graph(graph, nodelist, labels):
    fig, ax = plt.subplots(1, 1, figsize=(20, 20));
    nx.draw_networkx(graph, ax=ax, with_labels=False, nodelist=nodelist,
                     node_color=node_colors(labels), node_size=50)
                     

draw_bipartite_graph(train_graph, list(train_graph.nodes()), van_labels)


draw_bipartite_graph(train_graph, list(train_graph.nodes()), reg_labels)

# Bechmark the execution time for the computation of eigenvalues for the two matrices
print("Vanilla execution time: {}".format(benchmark(lambda: eig_laplacian(S))))
print("CoreCut execution time: {}".format(benchmark(lambda: eig_laplacian(SR))))


