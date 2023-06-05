from matplotlib import lines
import numpy as np
from array import array
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
# from node2vec import Node2Vec as n2v



G = nx.read_edgelist("../graph/karate.edgelist", nodetype=int)
embs = dict()
k = 0
index = []
full_list = []
total_vectors_in_file = 0
with open("../emb/karate.emb", 'r') as f:
  next(f)
  for i, line in enumerate(f):
    lst = line.split(' ')
    index.append(lst[0])
    lst.pop(0)
    list_of_integers = list(map(float, lst))
    # print(list_of_integers)
    full_list.append(list_of_integers)
    
    # print(A)
    # k = k + 1
    # if k == 3: 
    #   break    
A=np.array(full_list)
# print(np.array(full_list))

    # for line in myfile:
    #     temp = [result.add(item) for item in line.strip().split()]
    # jpg = line.strip().split(' ')
    # print(len(jpg))
    # print(jpg[1])
    #
clustering = SpectralClustering(
    n_clusters=5, 
    assign_labels='discretize',
    random_state=0
).fit(A)

comm_dct = dict(zip(index, clustering.labels_))

unique_coms = np.unique(list(comm_dct.values()))
cmap = {
    0 : 'maroon',
    1 : 'teal',
    2 : 'black', 
    3 : 'orange',
    4 : 'green',
}

# print(comm_dct)
node_cmap = [cmap[v] for _,v in comm_dct.items()]
# pos = nx.spring_layout(G)
# nx.draw(G, pos, node_size = 30, alpha = 0.8, node_color=node_cmap)
# plt.show()

comm_0 = []
comm_1 = []
comm_2 = []
comm_3 = []
comm_4 = []
for key in comm_dct:
  # print(key, '->', comm_dct[key])
  if comm_dct[key] == 0:
    comm_0.append(key)
  if comm_dct[key] == 1:
    comm_1.append(key)
  if comm_dct[key] == 2:
    comm_2.append(key)
  if comm_dct[key] == 3:
    comm_3.append(key)
  if comm_dct[key] == 4:
    comm_4.append(key)
value_lis = [comm_0,comm_1,comm_2,comm_3,comm_4]

comm = ["Community 0", "Community 1", "Community 2", "Community 3", "Community 4"]
dictt = {}
for i in range(0,len(value_lis)):
  dictt[comm[i]] = value_lis[i]

print(dictt)