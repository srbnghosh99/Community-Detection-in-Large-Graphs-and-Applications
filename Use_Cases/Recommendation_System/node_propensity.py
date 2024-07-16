# <#Title#>

import matplotlib.pyplot as plt
import pandas as pd
from networkx.readwrite import json_graph
import json
import networkx as nx
import seaborn as sns
from pathlib import Path
import csv
import ast
import subprocess
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import time
from os.path import dirname, join as pjoin


def node_propensity(dataset,cdfile,inputdirectory,outdirectory,overlap):
    directory = os.getcwd()
    cdfile = pjoin(directory,dataset, cdfile)
    inputdirectory = pjoin(directory,dataset, inputdirectory)
    outdirectory = pjoin(directory,dataset, outdirectory)
    
    detected_community_df = pd.read_csv(cdfile)
    closeness = []
    sameAsDegreeCentrality = []
    betweenness = []
    inCentrality = []
    outCentrality = []
    nodes = []
    community_name = []
    # print(detected_community_df)
    
    # detected_community_df['Node'] = detected_community_df['Node'].apply(ast.literal_eval)
    count = 0
    
    if(overlap == 'overlapping'):
        detected_community_df['Community'] = detected_community_df['Community'].apply(ast.literal_eval)
        for index, row in detected_community_df.iterrows():
            node = str(row['Node'])
            community = row['Community']
            a = []
            b = []
            c = []
            d = []
            e = []
            for comm in community:

                json_file = inputdirectory +  "/comm_" + str(comm) + ".json"
                # print(json_file)
                with open(json_file, 'r') as file:
                    data = json.load(file)
                    # print(data[node])
                    a.append(data[node]['closeness'])
                    b.append(data[node]['sameAsDegreeCentrality'])
                    c.append(data[node]['betweenness'])
                    d.append(data[node]['inCentrality'])
                    e.append(data[node]['outCentrality'])
            closeness.append(a)
            sameAsDegreeCentrality.append(b)
            betweenness.append(c)
            inCentrality.append(d)
            outCentrality.append(e)

        detected_community_df['Closeness'] = closeness
        detected_community_df['SameAsDegreeCentrality'] = sameAsDegreeCentrality
        detected_community_df['Betweenness'] = betweenness


        print(detected_community_df)
        filename = outdirectory + "/node_propensity_dataframe_test.csv"
        print(filename)
        detected_community_df.to_csv(filename,index = False)

    else:
        json_files = []

        # Iterate over the files in the directory
        for filename in os.listdir(inputdirectory):
            # Check if the file is a JSON file
            if filename.endswith('.json'):
                # Add the file to the list
                json_files.append(os.path.join(inputdirectory, filename))
        print(len(json_files))
        # json_file = inputdir +  "/comm_" + str(community) + ".json"
        for json_file in json_files:
            with open(json_file, 'r') as file:
                print(json_file)
                comm_value = os.path.splitext(os.path.basename(json_file))[0].split('_')[-1]

                # Convert the extracted value to an integer
                comm_value = int(comm_value)
                data = json.load(file)
                node_id_lis = list(data.keys())
                for node_id in node_id_lis:
                    node_data = data[node_id]
                    nodes.append(node_id)
                    community_name.append(comm_value)
                    closeness.append(node_data['closeness'])
                    sameAsDegreeCentrality.append(node_data['sameAsDegreeCentrality'])
                    betweenness.append(node_data['betweenness'])

        data = {
        'Node': nodes,
        'Community': community_name,
        'Closeness': closeness,
        'SameAsDegreeCentrality': sameAsDegreeCentrality,
        'Betweenness': betweenness
        }

        # Create a DataFrame from the dictionary
        df = pd.DataFrame(data)

        # Print the DataFrame
        print(df)
        oufile = outputdir + '/node_propensity_dataframe.csv'
        df.to_csv(oufile,index = False)
   
    


def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--dataset",type = str)
    parser.add_argument("--cdfile",type = str)
    parser.add_argument("--inputdir",type = str)
    parser.add_argument("--outputdir",type = str)
    parser.add_argument("--overlap",type = str)
    return parser.parse_args()

def main():
    start_time = time.time()
    inputs=parse_args()
    print(inputs.cdfile)
    print(inputs.inputdir)
    print(inputs.outputdir)
    node_propensity(inputs.dataset,inputs.cdfile,inputs.inputdir,inputs.outputdir,inputs.overlap)
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    elapsed_hours = int(elapsed_time_seconds // 3600)
    elapsed_minutes = int((elapsed_time_seconds % 3600) // 60)

    print("Elapsed Time:", elapsed_hours, "hours", elapsed_minutes, "minutes")
  

if __name__ == '__main__':
    main()

