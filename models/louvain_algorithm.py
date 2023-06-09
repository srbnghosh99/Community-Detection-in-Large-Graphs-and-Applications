from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import json
import csv
import pandas as pd

# load the karate club graph
G = nx.karate_club_graph()
G = nx.read_edgelist("merge_file_freq.edgelist", nodetype=str, data=(("weight", int),))


# compute the best partition
partition = community_louvain.best_partition(G)

# print(partition)
# file = 'louvain_authors.json' 
# json.dump( partition, open( "data/myfile.json", 'w' ) )
headerList = ['author','community']
with open('louvain_author_community.csv', 'w') as f:
    
    # dw = csv.DictWriter(f, delimiter=',', 
                        # fieldnames=headerList)
    for key in partition.keys():
        f.write("%s,%s\n"%(key,partition[key]))
df = pd.read_csv('test.csv')

df.columns = ['author','community']
df.to_csv("test.csv")
# with open(file, 'w') as f: 
#     json.dump(partition, f)
# draw the graph
# pos = nx.spring_layout(G)
# # color the nodes according to their partition
# cmap = cm.get_cmap('viridis', max(partition.values()) + 1)

# nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
#                        cmap=cmap, node_color=list(partition.values()))
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# plt.show()



