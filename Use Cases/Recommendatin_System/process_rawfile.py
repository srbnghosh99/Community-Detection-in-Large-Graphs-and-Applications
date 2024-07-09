from os.path import dirname, join as pjoin
import scipy.io as sio
import pandas as pd
import argparse
import sys
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# READ MAT FILES
#  Download the mat files from this link --> "https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm"


def raw_file_read(directory,dataset):

    # mat_fname = "/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/rating.mat"
    mat_fname = pjoin(directory,dataset, 'rating.mat')
    mat_contents = sio.loadmat(mat_fname)
    rating_array = mat_contents['rating']
    rating_df = pd.DataFrame(rating_array, columns=['userid', 'productid', 'categoryid', 'rating', 'helpfulness'])
    csv_fname = mat_fname.replace(".mat", ".csv")
    rating_df.to_csv(csv_fname,index = False)
    # mat_fname = "/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/trustnetwork.mat" 
    mat_fname = pjoin(directory,dataset, 'trustnetwork.mat')
    mat_contents = sio.loadmat(mat_fname)
    trustnetwork_array = mat_contents['trustnetwork']
    trustnet_df = pd.DataFrame(trustnetwork_array, columns=['user1', 'user2'])
    csv_fname = mat_fname.replace(".mat", ".csv")
    print(rating_df)
    print(trustnet_df)
    trustnet_df.to_csv(csv_fname,index = False)
    print("done")

    ### Renumber nodes
    G = nx.read_edgelist(csv_fname,delimiter=' ', nodetype=int)
    print(G.number_of_nodes(), G.number_of_edges(), len(sorted(G.nodes())))
    nodes_with_less_than_two_edges = [node for node, degree in G.degree() if degree < 2]
    print('number of user with less than two trustors',len(nodes_with_less_than_two_edges))
    G.remove_nodes_from(nodes_with_less_than_two_edges)
    df_edges = nx.to_pandas_edgelist(G)
    #df_edges.to_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/preprocessed_ciao_edges.csv",index=False, header=False)
    remaining_nodes = sorted(list(G.nodes()))
    mapping = {node: new_number + 1 for new_number, node in enumerate(remaining_nodes)}
    G_renumbered = nx.relabel_nodes(G, mapping)
    # Write the renumbered graph to a file
    output_file = "renumbered_graph_" + dataset + ".csv"
    nx.write_edgelist(G_renumbered, output_file, delimiter=' ', data=False)



def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--dataset",type = str)
    parser.add_argument("--directory",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    raw_file_read(inputs.directory,inputs.dataset)



if __name__ == '__main__':
    main()
