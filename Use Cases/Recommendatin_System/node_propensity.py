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
import subprocess#  <#Title#>




#def node_propensity():

closeness = []
sameAsDegreeCentrality = []
betweenness = []
detected_community_df = pd.read_csv("ciao_trustnet_ego_splitting_membership.csv")
print(detected_community_df)
detected_community_df['Community'] = detected_community_df['Community'].apply(ast.literal_eval)
for index, row in detected_community_df.iterrows():
    node = str(row['Node'])
    community = row['Community']
    #    print(detected_community_df)
#    print(community[0])
#    folder_path = "propensity"
#    file_names = os.listdir(folder_path)
#    print(len(file_names))
    a = []
    b = []
    c = []
    for comm in community:
        json_file = "propensity/comm_" + str(comm) + "_measure.json"
        with open(json_file, 'r') as file:
            data = json.load(file)
#            print(data)
#            print(data[node])
            a.append(data[node]['closeness'])
            b.append(data[node]['sameAsDegreeCentrality'])
            c.append(data[node]['betweenness'])
    closeness.append(a)
    sameAsDegreeCentrality.append(b)
    betweenness.append(c)
    

detected_community_df['Closeness'] = closeness
detected_community_df['SameAsDegreeCentrality'] = sameAsDegreeCentrality
detected_community_df['Betweenness'] = betweenness

print(detected_community_df)

detected_community_df.to_csv("node_propensity_dataframe.csv",index = False)
    
    
    
'''

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
'''
