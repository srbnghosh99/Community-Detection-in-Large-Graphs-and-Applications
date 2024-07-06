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
