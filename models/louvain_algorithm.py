from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import json
import csv
import pandas as pd
import os
import csv
import argparse
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def louvain_algo(inputfile,outputfile):
    # load the karate club graph
    #G = nx.karate_club_graph()
    # G = nx.read_edgelist("merge_file_freq.edgelist", nodetype=str, data=(("weight", int),))
    G = nx.read_edgelist(inputfile, nodetype=str, data=(("weight", int),))

    # compute the best partition
    partition = community_louvain.best_partition(G)

    # print(partition)
    # file = 'louvain_authors.json' 
    # json.dump( partition, open( "data/myfile.json", 'w' ) )
    headerList = ['author','community']
    with open('test.csv', 'w') as f:
        
        # dw = csv.DictWriter(f, delimiter=',', 
                            # fieldnames=headerList)
        for key in partition.keys():
            f.write("%s,%s\n"%(key,partition[key]))
    df = pd.read_csv('test.csv')

    df.columns = ['author','community']
    df.to_csv(outputfile, index = False, sep = ' ')
    df = df.sort_values(by=['author'])
    df1 = pd.read_csv("sorted_dblp_author_nameid.csv", sep = ' ')
    author_dic = dict(zip(df1.Id, df1.Author))
    df['author'] = df['author'].map(author_dic)
    df.community.value_counts().rename_axis('community').reset_index(name='counts').to_csv("louvain_comm_value_counts.csv", sep = ' ', index = False)
    

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

def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")

    parser.add_argument("--inputfilename",type = str)
    parser.add_argument("--outputfilename",type = str)
    
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.inputfilename)
    print(inputs.outputfilename)
    louvain_algo(inputs.inputfilename,inputs.outputfilename)
  

if __name__ == '__main__':
    main()


