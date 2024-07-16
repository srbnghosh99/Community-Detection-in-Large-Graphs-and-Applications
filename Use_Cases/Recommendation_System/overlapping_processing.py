import pandas as pd
import ast
from collections import OrderedDict
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from os.path import dirname, join as pjoin


def processing(dataset,inputfilename):
    
    directory = os.getcwd()
    inputfilename = pjoin(directory,dataset, inputfilename)
    df = pd.read_csv(inputfilename)
#    print(df)
    df['Community'] = df['Community'].apply(lambda x: list(set(ast.literal_eval(x))))
    list_of_lists = df['Community'].tolist()
    df['length_column'] = df['Community'].apply(lambda x: len(x))
    df = df.sort_values(by='length_column', ascending=False)
    flattened_list = [int(item) for sublist in list_of_lists for item in sublist]
    unique_list = set(flattened_list)
    len(unique_list)
    print(df)
    df.to_csv(inputfilename,index = False)
    community_mapping = {}
    for index, row in df.iterrows():
        nodes = row['Node']
        community = row['Community']
        # print(community)
        for c in community:
            # print(c)
            if c in community_mapping:
                community_mapping[c].append(nodes)
            else:
                community_mapping[c] = [nodes]

    community_mapping_df = pd.DataFrame(list(community_mapping.items()), columns=['Community', 'Nodes'])
    community_mapping_df['length_column'] = community_mapping_df['Nodes'].apply(lambda x: len(x))
    community_mapping_df = community_mapping_df.sort_values(by='length_column', ascending=False)
#    print(community_mapping_df)
    community_mapping_df.to_csv(inputfilename, index = False)
#    community_mapping_df = community_mapping_df.sort_values(by='Community', ascending=True)
#    community_mapping_df['Nodes'] = community_mapping_df['Nodes'].apply(ast.literal_eval)
#    community_mapping_df['Nodes'] = community_mapping_df['Nodes'].apply(lambda x: list(set(ast.literal_eval(x))))
#    community_mapping_df['length_column'] = community_mapping_df['Nodes'].apply(lambda x: len(x))
#    community_mapping_df = community_mapping_df.sort_values(by='length_column', ascending=True)
##    print(community_mapping_df)
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--dataset",type = str)
    parser.add_argument("--inputfilename",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.dataset)
    processing(inputs.dataset,inputs.inputfilename)

if __name__ == "__main__":
    main()
