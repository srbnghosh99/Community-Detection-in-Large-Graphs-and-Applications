import pandas as pd
import networkx as nx
from tqdm import tqdm
import ast
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse


#finan_comm
#
## Merge on 'Node' column
#merged_df = pd.merge(df_node_community, finan_comm, on='Node', how='outer', suffixes=('_df1', '_df2'))
#
#print(merged_df)
#
#
## Convert Community_df2 to list format
#merged_df['Community_df2'] = merged_df['Community_df2'].apply(lambda x: [x] if not isinstance(x, list) else x)
#
## Replace NaN in 'Community_df1' with 'Community_df2' and ensure 'Community' is in list format
#merged_df['Community'] = merged_df['Community_df1'].fillna(merged_df['Community_df2'])
#
#
### Drop the temporary columns
#merged_df = merged_df[['Node', 'Community']]
##
### Reset index
#merged_df = merged_df.reset_index(drop=True)
#
#print(merged_df)


#community_mapping = {}
#


    
#print(community_mapping)
#print(node_community_mapping)

# Feature 2: Ratio of communities for each node over its neighbors
def compute_community_ratio(node, graph, node_communities):
#    node_communities = set(community_dict.get(node, []))
    neighbors = list(graph.neighbors(node))
#    neighbor_communities = set()
#    for neighbor in neighbors:
#        neighbor_communities.update(community_dict.get(neighbor, []))
    
    if len(neighbors) == 0:
        return 0
    
    ratio = len(node_communities) / len(neighbors)
    return ratio

def compute_clique_score(node, graph, communities,community_mapping):
    clique_scores = []
    
    for c in communities:
        num_of_nodes = community_mapping[c]
        if len(num_of_nodes) < 2:
            # Skip communities with less than 2 nodes
            continue
        
        degree = sum(1 for neighbor in graph.neighbors(node) if neighbor in num_of_nodes)
        n = len(num_of_nodes)
        possible_edges = (n * (n - 1))/2
        
        # Check for division by zero
        if possible_edges > 0:
            clique_scores.append(degree / possible_edges)
    
    # If no valid communities, return 0 or some indicative value
    if not clique_scores:
        return 0
    
    avg_score = sum(clique_scores) / len(clique_scores)
    return avg_score

# Feature 4: Clique Score
#def compute_clique_score(node, graph, communities):
#    clique_score = []
#    for c in communities:
#        num_of_nodes = community_mapping[c]
#        degree = sum(1 for neighbor in graph.neighbors(node) if neighbor in num_of_nodes)
#        n = len(num_of_nodes)
#        possible_edges = (n * (n - 1))/2
#        clique_score.append(degree/possible_edges)
#    avg_score = sum(clique_score) / len(clique_score)
#    return avg_score



# Feature 5: Stark Score
def compute_stark_score(node, graph,communities,community_mapping):
    star_score = []
#    deg = graph.degree(node)
#    neighborlist = G.neighbors(node)
    
#    degree = sum(1 for neighbor in G.neighbors(node) if neighbor in community)
    for c in communities:
        num_of_nodes = community_mapping[c]
        degree = sum(1 for neighbor in graph.neighbors(node) if neighbor in num_of_nodes)
        n = len(num_of_nodes)
#        subgraph = graph.subgraph(num_of_nodes)
#        num_edges = subgraph.number_of_edges()
#        possible_edges = (num_edges * (num_edges - 1) )
        possible_edges = n * (n - 1)
        star_score.append(degree/possible_edges)
    avg_score = sum(star_score) / len(star_score)
#    neighbors = list(graph.neighbors(node))
#    num_neighbors = len(neighbors)
#    if num_neighbors == 0:
#        return 0
#
#    total_edges = sum(len(list(graph.neighbors(neighbor))) for neighbor in neighbors)
#    avg_degree = total_edges / num_neighbors
#
#    # Stark score could be defined in multiple ways. Here, we use a simple deviation from average degree
#    deviation_from_avg = abs(len(neighbors) - avg_degree)
    
    return avg_score

def find_auxiliary_communities(graph_file,auxiliary_community_file,output):

    G = nx.read_edgelist(graph_file,delimiter=' ',nodetype=int)
    print(G.number_of_nodes())

    df_node_community = pd.read_csv(auxiliary_community_file)
    df_node_community['Node'] = df_node_community['Node'].apply(lambda x: list(set(ast.literal_eval(x))))
    #print(df_node_community)
    merged_df = df_node_community
    # finan_comm = pd.read_csv("/Users/shrabanighosh/Downloads/data/outlier_detection/YelpHotel/spectral.csv",sep = ',')

    community_mapping = df_node_community.set_index('Community')['Node'].to_dict()
    print(merged_df)
    node_community_mapping = {}
    for index, row in merged_df.iterrows():
        nodes = row['Node']
        community = row['Community']
        # print(community)
        for n in nodes:
            # print(c)
            if n in node_community_mapping:
                node_community_mapping[n].append(community)
            else:
                node_community_mapping[n] = [community]
    #        print(node_community_mapping)
    #        break
    column_names = ['Node','Community']
    merged_df = pd.DataFrame(list(node_community_mapping.items()), columns=column_names)

    print(merged_df)


    num_communities = []
    clustering_coeff = []
    clique_score = {}
    stark_score = {}
    for index,row in merged_df.iterrows():
        node = row['Node']
        comm_lis = row['Community']
        num_communities.append(len(comm_lis))
        community_ratio = {node: compute_community_ratio(node, G, comm_lis) for node in G.nodes()}
        clique_score[node] = compute_clique_score(node, G,comm_lis,community_mapping)
        stark_score[node] = compute_stark_score(node, G,comm_lis,community_mapping)
        
    # Feature 3: Clustering coefficient of each node
    clustering_coefficients = nx.clustering(G)
    for node, coeff in clustering_coefficients.items():
    #    print(f"Node {node}: Clustering Coefficient {coeff}")
    #        clustering_coeff[node] = coeff
            clustering_coeff.append(coeff)
            
#clique_score = {node: compute_clique_score(node, G) for node in G.nodes()}
    
#stark_score = {node: compute_stark_score(node, G) for node in G.nodes()}

#print(len(num_communities),len(clustering_coeff))

## Combine features into a DataFrame
    df_features = pd.DataFrame({
        'Node': list(G.nodes()),
        'Num_Communities': num_communities,
        'Community_Ratio': [community_ratio[node] for node in G.nodes()],
        'Clustering_Coeff': clustering_coeff,
        'Clique_Score': [clique_score[node] for node in G.nodes()],
        'Stark_Score': [stark_score[node] for node in G.nodes()]
    })

    # Display the DataFrame
    print(df_features)
    #
    df_features.to_csv(output, index = False)

def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")
    
    parser.add_argument("--graph_file",type = str)
    parser.add_argument("--auxiliary_community_file",type = str)
    parser.add_argument("--outputfile",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.graph_file)
    print(inputs.auxiliary_community_file)
    find_auxiliary_communities(inputs.graph_file,inputs.auxiliary_community_file,inputs.outputfile)
  
if __name__ == '__main__':
    main()