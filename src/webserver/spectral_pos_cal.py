import matplotlib.pyplot as plt
import networkx as nx
import json 
import igraph as ig
import argparse
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def spectral_pos_cal(filename):
    G = nx.read_edgelist(filename, delimiter=" ", data=(("Weight", int),))
    print(G.number_of_nodes(),G.number_of_edges())

    pos = nx.spectral_layout(G, dim = 3)



    pts_final_dict = {k:v.tolist() for k,v in pos.items()}
    out_file = open("spectral_layout.json", "w") 
    json.dump(pts_final_dict, out_file, indent = 3) 
    out_file.close() 


    H = ig.Graph.from_networkx(G)
    layt=H.layout('kk', dim=3)

    N = G.number_of_nodes()
    Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[layt[k][1] for k in range(N)]# y-coordinates
    Zn=[layt[k][2] for k in range(N)]# z-coordinates
    # print(Xn,Yn,Zn)

    pts_final_dict={}
    for i in range(0,len(layt)):
        pts_final_dict[i] = layt[i]
    with open("igraph_layout.json", "w") as outfile:
        json.dump(pts_final_dict, outfile)
    
def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")

    parser.add_argument("--filename",type = str)
    
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.filename)
    spectral_pos_cal(inputs.filename)





