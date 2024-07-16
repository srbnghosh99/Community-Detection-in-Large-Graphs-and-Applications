#  <#Title#>

import json
import pandas as pd
import os
import argparse
import sys
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import random 

def center_of_cluster(dataset):
    features = []
    clusters = []
    nodes = []
    values = []
    outfile = dataset+ '_ClusterCenter.csv'
    file_path = dataset+ '_ClusterCenter.txt'
    folder_list = ['betweenness','inCentrality','outCentrality','sameAsDegreeCentrality']
    with open(file_path, 'w') as file:
        for folder_name in folder_list:
            file.write(folder_name+'\n')
            folder_path = dataset + '/' + folder_name + '/'
            print(folder_path)
            file_names = os.listdir(folder_path)
            # Iterate over the file names
            for file_name in file_names:
                # Print each file name
                if ".DS_Store" not in file_name:
#                    cluster = re.search(r'\d+', file_name)
                    numeric_value = re.findall(r'\d+', file_name)
                    cluster_name = int(numeric_value[0])
#                    print(cluster)
                    file_name =folder_path +file_name
                    print(file_name)

                    with open(file_name,'r') as json_file:
                        data = json.load(json_file)
                        # df = pd.read_json(data)
                        df = pd.DataFrame(list(data.items()), columns=['Node', 'Value'])
                        max_row_index = df['Value'].idxmax()
                        max_row = df.loc[max_row_index]
                        print('Center of Cluster',max_row['Node'])
                        features.append(folder_name)
                        clusters.append(cluster_name)
                        nodes.append(max_row['Node'])
                        values.append(df['Value'].max())
    data = {
    'feature': features,
    'cluster': clusters,
    'Node': nodes,
    'Value':values
    }
    df = pd.DataFrame(data)
    df.to_csv(outfile)
#                        file.write('Center of Cluster {}: {}\n'.format(cluster,max_row['Node']))

# def trust_prediction(inputfile):

                    
def center_cluster_calculate(directory):
    print()
    center_of_cluster = {}
    files = os.listdir(directory)  
    clusters = []
    max_closeness_nodes = []
    random_nodes = []
    max_sameAsDegreeCentrality_nodes = []
    max_betweenness_nodes = []
    max_inCentrality_nodes = []
    max_outCentrality_nodes = []
    for filename in files:
        if ".DS_Store" in filename:
            continue
        if os.path.isfile(os.path.join(directory, filename)):
           
            # Full path to the file
            input_file_name = os.path.join(directory, filename)     
            # Read the JSON file
            with open(input_file_name, 'r') as file:
                data = json.load(file)
            print("cluster: ", filename)
            cluster = filename.split('.')[0]
            cluster = cluster.split('_')[1]
            clusters.append(cluster)
            # Initialize variables to store the highest values and corresponding nodes
            max_closeness_node = None
            max_sameAsDegreeCentrality_node = None
            max_betweenness_node = None
            max_inCentrality_node = None
            max_outCentrality_node = None
            max_closeness = float('-inf')
            max_sameAsDegreeCentrality = float('-inf')
            max_betweenness = float('-inf')
            max_inCentrality = float('-inf')
            max_outCentrality = float('-inf')

#            print(list(data.items()))
            # Randomly select a node
            random_node = random.choice(list(data.items()))
#            print(random_node[0])
            random_nodes.append(random_node[0])
            for node, attributes in data.items():
                closeness = attributes.get("closeness", 0)
                sameAsDegreeCentrality = attributes.get("sameAsDegreeCentrality", 0)
                betweenness = attributes.get("betweenness", 0)
                inCentrality = attributes.get("inCentrality", 0)
                outCentrality = attributes.get("outCentrality", 0)

                # Update variables if a higher value is found
                if closeness > max_closeness:
                    max_closeness = closeness
                    max_closeness_node = node
                    

                if sameAsDegreeCentrality > max_sameAsDegreeCentrality:
                    max_sameAsDegreeCentrality = sameAsDegreeCentrality
                    max_sameAsDegreeCentrality_node = node
                    

                if betweenness > max_betweenness:
                    max_betweenness = betweenness
                    max_betweenness_node = node
                    

                if inCentrality > max_inCentrality:
                    max_inCentrality = inCentrality
                    max_inCentrality_node = node
                    
                if outCentrality > max_outCentrality:
                    max_outCentrality = outCentrality
                    max_outCentrality_node = node
                    
            max_closeness_nodes.append(max_closeness_node)
            max_sameAsDegreeCentrality_nodes.append(max_sameAsDegreeCentrality_node)
            max_betweenness_nodes.append(max_betweenness_node)
            max_inCentrality_nodes.append(max_inCentrality_node)
            max_outCentrality_nodes.append(max_outCentrality_node)

#            print(len(clusters),len(max_closeness_nodes),len(max_sameAsDegreeCentrality_nodes),
#          len(max_outCentrality_nodes),len(max_inCentrality_nodes),len(random_nodes))
        #     print(clusters,max_closeness_nodes, max_inCentrality_nodes )
            # Print the results
            print(f"Highest closeness: {max_closeness} (Node: {max_closeness_node})")
            print(f"Highest sameAsDegreeCentrality: {max_sameAsDegreeCentrality} (Node: {max_sameAsDegreeCentrality_node})")
            print(f"Highest betweenness: {max_betweenness} (Node: {max_betweenness_node})")
            print(f"Highest inCentrality: {max_inCentrality} (Node: {max_inCentrality_node})")
            print(f"Highest outCentrality: {max_outCentrality} (Node: {max_outCentrality_node})")
    
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Cluster': clusters,
        'MaxClosenessNode': max_closeness_nodes,
        'MaxSameAsDegreeCentralityNode': max_sameAsDegreeCentrality_nodes,
        'MaxBetweennessNode': max_betweenness_nodes,
        'MaxOutCentralityNode': max_outCentrality_nodes,
        'MaxinCentralityNode': max_inCentrality_nodes,
        'RandomNode': random_nodes
    })
    print(df)
    outfile = directory + '/centerclusters.csv'
    print(outfile)
    df.to_csv(outfile)
    # trust_prediction(outfile)

    
            

def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--directory",type = str)
    # parser.add_argument("--outfile",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.directory)
    # print(input.outfile)
    center_cluster_calculate(inputs.directory)

if __name__ == '__main__':
    main()



#code run 
# python find_center_of_communities.py --directory /Users/shrabanighosh/Downloads/data/trust_prediction/ciao/propensity_subgraph
