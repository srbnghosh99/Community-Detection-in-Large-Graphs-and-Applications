import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
from networkx.readwrite import json_graph
import json
import networkx as nx
import seaborn as sns
from pathlib import Path
import csv
import ast
import subprocess# 
import os
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import shutil



closeness = []
sameAsDegreeCentrality = []
betweenness = []
inCentrality = []
outCentrality = []



def clear_folder(outdirectory):
    # Check if the folder exists
    if os.path.exists(outdirectory):
        # Remove all files in the folder
        for filename in os.listdir(outdirectory):
            file_path = os.path.join(outdirectory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"The folder {outdirectory} does not exist.")

def node_propensity(inDirectory,outdirectory):


    # List of folder names you want to create
    # folder_names = ['betweenness', 'MaxDegree', 'MaxTrustor', 'MaxTrustee','Random']

    # # # Create the folders
    # for folder_name in folder_names:
    #     current_directory = os.getcwd()
    #     folder_path = os.path.join(current_directory, folder_name)
        
    #     # Check if the folder already exists before creating it
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #         print(f"Folder '{folder_name}' created successfully at '{folder_path}'.")
    #     else:
    #         print(f"Folder '{folder_name}' already exists at '{folder_path}'.")

    files = os.listdir(inDirectory)
    clear_folder(outdirectory)

    # Iterate through the files
    for file_name in files:
        # Check if the item is a file (not a directory)
        if os.path.isfile(os.path.join(inDirectory, file_name)):
            # Full path to the file
            input_file_name = os.path.join(inDirectory, file_name)
            # with open(json_file, 'r') as file:
            #     data = json.load(file)
            node_script_path = '/Users/shrabanighosh/Downloads/ngraph.centrality-main/myscript_copy.js'
            # Get the directory of the Node.js script
            script_directory = os.path.dirname(os.path.abspath(node_script_path))
            print("Script Directory:", script_directory)
            # Replace 'file_name.json' and 'output_file.json' with your actual parameters
    
            output_file_name = outdirectory +'/' +file_name

            # Construct the command to run the Node.js script
            command = ['node', node_script_path, input_file_name, output_file_name]

            # Run the command
            try:
                subprocess.run(command, check=True, cwd=script_directory)
                print(f"Node.js script executed successfully for {input_file_name}.")
            except subprocess.CalledProcessError as e:
                print(f"Error executing Node.js script: {e}")


    # detected_community_df = pd.read_csv("ciao_trustnet_ego_splitting_membership.csv")
    # print(detected_community_df)
    # detected_community_df['Community'] = detected_community_df['Community'].apply(ast.literal_eval)
    # for index, row in detected_community_df.iterrows():
    #     node = str(row['Node'])
    #     community = row['Community']
    #     #    print(detected_community_df)
    # #    print(community[0])
    # #    folder_path = "propensity"
    # #    file_names = os.listdir(folder_path)
    # #    print(len(file_names))
    #     a = []
    #     b = []
    #     c = []
    #     for comm in community:
    #         json_file = "propensity/comm_" + str(comm) + "_measure.json"
    #         with open(json_file, 'r') as file:
    #             data = json.load(file)
    # #            print(data)
    # #            print(data[node])
    #             a.append(data[node]['closeness'])
    #             b.append(data[node]['sameAsDegreeCentrality'])
    #             c.append(data[node]['betweenness'])
    #     closeness.append(a)
    #     sameAsDegreeCentrality.append(b)
    #     betweenness.append(c)
        

    # detected_community_df['Closeness'] = closeness
    # detected_community_df['SameAsDegreeCentrality'] = sameAsDegreeCentrality
    # detected_community_df['Betweenness'] = betweenness

    # print(detected_community_df)

    # detected_community_df.to_csv("node_propensity_dataframe.csv",index = False)
    
    
    


def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--inDirectory",type = str)
    parser.add_argument("--outDirectory",type = str)
#    parser.add_argument("--outputfilename",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.inDirectory)
    print(inputs.outDirectory)
    node_propensity(inputs.inDirectory,inputs.outDirectory)
  

if __name__ == '__main__':
    main()

#code run 
#    
# python3 node_propensity.py --inDirectory /Users/shrabanighosh/Downloads/data/trust_prediction/ciao/louvain
#     --outDirectory /Users/shrabanighosh/Downloads/data/trust_prediction/ciao/propensity_subgraph

