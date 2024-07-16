import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
from networkx.readwrite import json_graph
import json
import networkx as nx
import seaborn as sns
from pathlib import Path
import csv
from tqdm import tqdm


G = nx.read_edgelist("/Users/shrabanighosh/Downloads/data/outlier_detection/DGraph_fin_header_false.edgelist",delimiter=' ',nodetype=int)
print(G.number_of_nodes(), G.number_of_edges())

finan_comm = pd.read_csv("/Users/shrabanighosh/Downloads/data/outlier_detection/assignments/DGraph_fin_header_false_label_prop.csv",sep = ',')
# finan_comm.columns = ['Node','Community']
# finan_comm.sort_values(by=['Community'], key=lambda x: x.astype(int)).reset_index()

finan_comm = finan_comm.sort_values(by=['Node'], key=lambda x: x.astype(int)).reset_index()
print(finan_comm)


count = 0
auxiliary_communities = []

def are_all_same(lst):
    return all(x == lst[0] for x in lst)

total_nodes = len(G.nodes())
progress_bar = tqdm(total=total_nodes, desc="Processing Nodes", unit="node")

for node in G.nodes(data=False):
    # print(node)
    node_comm = finan_comm.loc[finan_comm['Node'] == node, 'Community'].iloc[0]
    # print(finan_comm.loc[finan_comm['Node'] == node])
    # print("node: ",node, node_comm) 
    # break
    neighbors = list(G.neighbors(node))
    df = finan_comm[finan_comm['Node'].isin(neighbors)]
    comm = df["Community"].tolist()
    result_different = are_all_same(comm)
    if result_different == False:
        auxiliary_communities.append(node_comm)
    progress_bar.update()
progress_bar.close()
print("DOne")
print(len(auxiliary_communities))

