import pandas as pd
import networkx as nx
from tqdm import tqdm
import ast
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
import numpy as np



def community_cliqueness(G,Node_Communitylis,Community_Nodelis,str,end):
    num_communities = {}
    clique_score = {}
    stark_score = {}
    nodelist = []
    community_ratio = {}
    nodelist = list(range(str, end))
    # for node in tqdm(list(G.nodes())):
    overlap = 'overlapping'
    if overlap == 'nonoverlapping':
        for node in tqdm(nodelist):
            clique_scores = []
            stark_scores = []
            # print()
            neighbors = list(G.neighbors(node))
            node_comm = 1
            # node_comm = Node_Communitylis[Node_Communitylis['Node'] == node]['Community'].iloc[0]
            # print(node_comm)
            # number_of_communities = Node_Communitylis['Community'].nunique()
            number_of_communities = 1
            ratio = (number_of_communities) / len(neighbors)
            set1 = set(neighbors)
            # num_of_nodes = Community_Nodelis[Community_Nodelis['Community'] == node_comm]['Node'].iloc[0]
            num_of_nodes = nodelist
            print(len(num_of_nodes))
            set2 = set(num_of_nodes)
            common_nodes = list(set1.intersection(set2))
            common_nodes.append(node)
            n = len(common_nodes)
            H = G.subgraph(common_nodes).copy()
            H.remove_edges_from(nx.selfloop_edges(H))
            degree = sum(H.degree(neighbor) for neighbor in H.neighbors(node))
            num_edges = H.number_of_edges()
            possible_edges = (n * (n - 1)) / 2
            if possible_edges > 0:
                cliqueness = num_edges / possible_edges
                clique_scores.append(cliqueness)
                stark_scores.append(H.degree(node) / (degree + 1))
            else:
                clique_scores.append(0)
                stark_scores.append(0)
            num_communities[node] = number_of_communities
            community_ratio[node] = ratio
            clique_score[node] = np.average(clique_scores)
            stark_score[node] = np.average(stark_scores)
        return nodelist, num_communities, community_ratio, clique_score, stark_score
    # print("here")
    for node in tqdm(nodelist):
        clique_scores = []
        stark_scores = []
        neighbors = list(G.neighbors(node))
        node_comm = Node_Communitylis[Node_Communitylis['Node'] == node]['Community'].iloc[0]
        if isinstance(node_comm, list):
            print("1")
            for c in node_comm:
                num_of_nodes = Community_Nodelis[Community_Nodelis['Community'] == c]['Node'].iloc[0]
                number_of_communities = len(node_comm)
                ratio = (number_of_communities) / len(neighbors)
                set2 = set(num_of_nodes)
                # print(neighbors,num_of_nodes)
                common_nodes = list(set1.intersection(set2))
                # print('node',node)
                common_nodes.append(node)
                n = len(common_nodes)
                # print('common_nodes',common_nodes)
                # degree = sum(1 for neighbor in H.neighbors(node) if neighbor in common_nodes)
                # num_nodes = G.degree(node)
                H = G.subgraph(common_nodes).copy()
                H.remove_edges_from(nx.selfloop_edges(H))
                # print(H.nodes())
                # print(H.degree(node))
                degree = sum(H.degree(neighbor) for neighbor in H.neighbors(node))
                # print('degree,',degree)
                num_edges = H.number_of_edges()
                # print(H.edges())
                # print('n',n)
                possible_edges = (n * (n - 1)) / 2
                if possible_edges > 0:
                    cliqueness = num_edges / possible_edges
                    clique_scores.append(cliqueness)
                    stark_scores.append(H.degree(node) / (degree + 1))
                else:
                    clique_scores.append(0)
                    stark_scores.append(0)
        else:
            # print(node)
            # print("---------------------")
            number_of_communities = 1
            ratio = (number_of_communities) / len(neighbors)
            set1 = set(neighbors)
            num_of_nodes = Community_Nodelis[Community_Nodelis['Community'] == node_comm]['Node'].iloc[0]
            # print(num_of_nodes)
            set2 = set(num_of_nodes)
            common_nodes = list(set1.intersection(set2))
            n = len(common_nodes)
            common_nodes.append(node)
            H = G.subgraph(common_nodes).copy()
            H.remove_edges_from(nx.selfloop_edges(H))
            degree = sum(H.degree(neighbor) for neighbor in H.neighbors(node))
            num_edges = H.number_of_edges()
            possible_edges = (n * (n - 1)) / 2
            if possible_edges > 0:
                cliqueness = num_edges / possible_edges
                clique_scores.append(cliqueness)
                stark_scores.append(H.degree(node) / (degree + 1))
            else:
                clique_scores.append(0)
                stark_scores.append(0)
        num_communities[node] = number_of_communities
        community_ratio[node] = ratio
        clique_score[node] = np.average(clique_scores)
        stark_score[node] = np.average(stark_scores)
    # print(nodelist, num_communities, community_ratio, clique_score, stark_score)
    return nodelist, num_communities, community_ratio, clique_score, stark_score

def find_features(graph_file,communitymap_file,nodemap_file, output,startnode,endnode):
    
    G = nx.read_edgelist(graph_file,delimiter=' ',nodetype=int)
    print(G.number_of_nodes(),G.number_of_edges())
    Node_Communitylis = pd.read_csv(communitymap_file)
    Community_Nodelis = pd.read_csv(nodemap_file)
    print(Community_Nodelis)
    print(Node_Communitylis)
    nodelist, num_communities,community_ratio,clique_score,stark_score = community_cliqueness(G,Node_Communitylis,Community_Nodelis,startnode,endnode)
    # print(nodelist,num_communities,community_ratio,clique_score,stark_score)
    df_features = pd.DataFrame({
        'Node': nodelist,
        'Num_Communities': [num_communities[node] for node in nodelist],
        'Community_Ratio': [community_ratio[node] for node in nodelist],
       # 'Clustering_Coeff': [clustering_coefficients[node] for node in G.nodes()],
        'Clique_Score': [clique_score[node] for node in nodelist],
        'Stark_Score': [stark_score[node] for node in nodelist]
    })

    print(df_features)
    df_features.to_csv(output, index = False)

def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")
    
    parser.add_argument("--graph_file",type = str)
    parser.add_argument("--communitymap_file",type = str) #CommunitymaptoNoderame[community:int][nodelist:int]
    parser.add_argument("--nodemap_file",type = str)      #NodemaptoCommunity[node:int][communitylist:int]
    parser.add_argument("--outputfile",type = str)
    parser.add_argument("--startnode",type = int)
    parser.add_argument("--endnode",type = int)
    return parser.parse_args()

def main():
    inputs=parse_args()
    # print(inputs.graph_file)
    # print(inputs.community_file)
    find_features(inputs.graph_file,inputs.communitymap_file,inputs.nodemap_file,inputs.outputfile,inputs.startnode,inputs.endnode)
  
if __name__ == '__main__':
    main()

# python3 feature_extraction_copy.py --graph_file Financial_data/DGraphFin/dgfingraph_space.csv --communitymap_file Financial_data/DGraphFin/assignments/dgfingraph_ego.csv --nodemap_file Financial_data/DGraphFin/assignments/dgfingraph_ego2.csv --outputfile Financial_data/DGraphFin/assignments/dgfingraph_ego_features1.csv --startnode 740110 --endnode 1480220
