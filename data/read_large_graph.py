import pandas as pd
import networkx as nx


df = pd.read_csv("myout1.edgelist",sep = ' ')
df.sort_values(by='freq', ascending=False)
df.sort_values(by='freq', ascending=False).to_csv("sorted_dblp_header_false.edgelist", header = False, index = False, sep = ' ')
G = nx.read_edgelist("sorted_dblp_header_false.edgelist", nodetype=str, data=(("weight", int),))
G.number_of_nodes(),G.number_of_edges()
# df1 = df.sort_values(by='freq', ascending=False)
df1 = df[df["freq"]> 1].shape
df1.sort_values(by='freq', ascending=False).to_csv("sorted_freq_ge_1_header_false.edgelist", index = False, sep = ' ')

g = nx.read_edgelist("sorted_freq_ge_1_header_false.edgelist", nodetype=str, data=(("weight", int),))
g.number_of_nodes(),g.number_of_edges()
