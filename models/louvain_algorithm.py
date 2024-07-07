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
    G=nx.read_weighted_edgelist(inputfile)
    # compute the best partition
    partition = community_louvain.best_partition(G)
    # print(partition)

    with open('test.csv', 'w') as f:
        
        # dw = csv.DictWriter(f, delimiter=',', 
                            # fieldnames=headerList)
        for key in partition.keys():
            f.write("%s,%s\n"%(key,partition[key]))
    column_names = ['Node','Community']
    df = pd.read_csv('test.csv',names = column_names)
    df = df.sort_values(by=['Node'])
    print(df)

    df.to_csv(outputfile, sep = ',',index = False)


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
