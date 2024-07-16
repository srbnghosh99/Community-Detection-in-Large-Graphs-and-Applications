#  <#Title#>


import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
from networkx.readwrite import json_graph
import json
import networkx as nx
import seaborn as sns
from pathlib import Path
import csv
import matplotlib.pyplot as plt  # Optional, for plotting
import ast
import argparse
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter



def preprocessing(filename):
    G = nx.read_edgelist(filename,delimiter=' ', nodetype=int)
    print(G.number_of_nodes(), G.number_of_edges(), len(sorted(G.nodes())))
    nodes_with_less_than_two_edges = [node for node, degree in G.degree() if degree < 2]
    print('number of user with less than two trustors',len(nodes_with_less_than_two_edges))
    G.remove_nodes_from(nodes_with_less_than_two_edges)
    df_edges = nx.to_pandas_edgelist(G)
    df_edges.to_csv('preprocessed_graph_edges.csv', index=False, header=False)



def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--inputfilename",type = str)
    parser.add_argument("--overlapping",type = str)
#    parser.add_argument("--outputfilename",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.inputfilename)
    print(inputs.overlapping)
    community_vis(inputs.inputfilename,inputs.overlapping)
  

if __name__ == '__main__':
    main()
