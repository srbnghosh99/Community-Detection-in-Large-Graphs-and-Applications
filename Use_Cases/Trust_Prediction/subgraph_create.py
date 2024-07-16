import networkx as nx
import seaborn as sns
from pathlib import Path
import csv
import matplotlib.pyplot as plt  # Optional, for plotting
import ast
import argparse
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import json
from networkx.readwrite import json_graph


def find_subgraphs(graphfile, filename,overlapping,outdirectory):

    G = nx.read_edgelist(graphfile,delimiter=' ', nodetype=int)
    print(f"Number of nodes {G.number_of_nodes()}, Number of edges {G.number_of_edges()}")
    if (overlapping == 'nonoverlapping'):
        trustnet = pd.read_csv(filename,sep = ',')
        print(trustnet)
        print("Number of communities", trustnet['Community'].nunique())
        # trustnet = pd.DataFrame(data.items(), columns=['Node', 'Community'])
        trustnet['Community'].value_counts()
        freqdf = trustnet['Community'].value_counts().reset_index()
        print(freqdf)
        communitilist = freqdf['Community'].tolist()
        print(communitilist)
        for i in communitilist:
            print(i)
            nodes_to_include = trustnet[trustnet['Community'] == i]['Node'].tolist()
            print('nodes_to_include',len(nodes_to_include))
            # Create a subgraph from the list of nodes
#            del subgraph
            subgraph = G.subgraph(nodes_to_include)
            print('nodes {} edges {}'.format(subgraph.number_of_nodes(),subgraph.number_of_edges()))
            # subgraph.number_of_edges(),subgraph.number_of_nodes()
            json_data = json_graph.node_link_data(subgraph, {'source': 'fromId', 'target': 'toId'})
            outputfile = outdirectory + "/comm_"+ str(i)+ ".json"
            with open(outputfile,'w') as json_file:
                json.dump(json_data,json_file,separators=(',', ':'))
        
    else:
        p = Path(filename)
        with p.open('r', encoding='utf-8') as f:
            data = json.loads(f.read())
        # print(data)
        trustnet_egosplit = pd.DataFrame(data.items(), columns=['Node', 'Community'])
        trustnet_egosplit['Node'] = trustnet_egosplit['Node'].astype(int)
        trustnet_egosplit = trustnet_egosplit.sort_values(by=['Node']).reset_index()
        trustnet_egosplit = trustnet_egosplit.drop('index', axis=1)
        print(trustnet_egosplit)
        community_dict = {}
        community_mapping = {}
        for index, row in trustnet_egosplit.iterrows():
            nodes = row['Node']
            community = row['Community']
            # print(community)
            for c in community:
                # print(c)
                if c in community_mapping:
                    community_mapping[c].append(nodes)
                else:
                    community_mapping[c] = [nodes]

        # for index, row in trustnet_egosplit.iterrows():
        #     node = row['Node']
        #     community = row['Community']
        #     # print(node, community)
        #     for comm_id in community:
        #         if comm_id in community_dict:
        #             existing_value = community_dict[comm_id]
        #             if isinstance(existing_value, int):
        #                 community_dict[comm_id] = [existing_value]
        #             else:
        #                 lis = community_dict[comm_id]
        #                 lis.append(node)
        #                 community_dict[comm_id] = lis
        #         else:
        #             community_dict[comm_id] = node
        #         # print(community_dict)
        #     # break
        community_mapping_df = pd.DataFrame(list(community_mapping.items()), columns=['Community', 'Nodes'])
        print(community_mapping_df)

        print("Number of communities", len(community_mapping_df))
        for key in community_mapping:
            # nodes_to_include = trustnet_egosplit[trustnet_egosplit['Community'] == key]['Node'].tolist()
            
            nodes_to_include = community_mapping[key]
            subgraph = G.subgraph(nodes_to_include)
            # subgraph.number_of_edges(),subgraph.number_of_nodes()
            json_data = json_graph.node_link_data(subgraph, {'source': 'fromId', 'target': 'toId'})
            outputfile = outdirectory + "/comm_"+ str(key)+ ".json"
            with open(outputfile,'w') as json_file:
                json.dump(json_data,json_file,separators=(',', ':'))



def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--graphfile",type = str)
    parser.add_argument("--inputfilename",type = str)
    parser.add_argument("--overlapping",type = str)
    parser.add_argument("--outdirectory",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.graphfile)
    print(inputs.inputfilename)
    print(inputs.overlapping)
    print(inputs.outdirectory)
    # community_vis(inputs.inputfilename,inputs.overlapping)
    find_subgraphs(inputs.graphfile,inputs.inputfilename,inputs.overlapping,inputs.outdirectory)

  

if __name__ == '__main__':
    main()


## code run
    '''
python3 subgraph_create.py --graphfile /Users/shrabanighosh/Downloads/data/trust_prediction/epinions/renumbered_graph_epinions.csv --inputfilename /Users/shrabanighosh/Downloads/data/trust_prediction/epinions/renumbered_graph_epinions_label_prop.csv --overlapping nonoverlapping --outdirectory /Users/shrabanighosh/Downloads/data/trust_prediction/epinions/label_propagation

python3 subgraph_create.py --graphfile /Users/shrabanighosh/Downloads/data/trust_prediction/ciao/renumbered_graph_ciao.csv --inputfilename /Users/shrabanighosh/Downloads/data/trust_prediction/ciao/label_propagation_ciao_trustnet.csv --overlapping nonoverlapping --outdirectory /Users/shrabanighosh/Downloads/data/trust_prediction/ciao/label_propagation
'''
