#  <#Title#>

import pandas as pd
import networkx as nx
#trustnet


G = nx.read_edgelist("/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/ciao_trustnet.csv",delimiter=' ', nodetype=int)
print(G.number_of_nodes(), G.number_of_edges(), len(sorted(G.nodes())))
nodes_with_less_than_two_edges = [node for node, degree in G.degree() if degree < 2]
print('number of user with less than two trustors',len(nodes_with_less_than_two_edges))
G.remove_nodes_from(nodes_with_less_than_two_edges)
df_edges = nx.to_pandas_edgelist(G)
df_edges.to_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/preprocessed_ciao_edges.csv",index=False, header=False)
remaining_nodes = sorted(list(G.nodes()))
print(remaining_nodes[0:10])
mapping = {node: new_number + 1 for new_number, node in enumerate(remaining_nodes)}
#print(mapping)
G_renumbered = nx.relabel_nodes(G, mapping)
# Write the renumbered graph to a file
output_file = "renumbered_graph_ciao.csv"
nx.write_edgelist(G_renumbered, output_file, delimiter=' ', data=False)


G = nx.read_edgelist("/Users/shrabanighosh/Downloads/data/trust_prediction/epinions/epinions_trustnet.csv",delimiter=' ', nodetype=int)
print(G.number_of_nodes(), G.number_of_edges(), len(sorted(G.nodes())))
nodes_with_less_than_two_edges = [node for node, degree in G.degree() if degree < 2]
print('number of user with less than two trustors',len(nodes_with_less_than_two_edges))
G.remove_nodes_from(nodes_with_less_than_two_edges)
df_edges = nx.to_pandas_edgelist(G)
df_edges.to_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/epinions/preprocessed_epinions_edges.csv",index=False, header=False)
remaining_nodes = sorted(list(G.nodes()))
print(remaining_nodes[0:10])
mapping = {node: new_number + 1 for new_number, node in enumerate(remaining_nodes)}
#print(mapping[0:10])
print(mapping)
G_renumbered = nx.relabel_nodes(G, mapping)
# Write the renumbered graph to a file
output_file = "renumbered_graph_epinions.csv"
nx.write_edgelist(G_renumbered, output_file, delimiter=' ', data=False)


'''
ciao_trust = pd.read_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/ciao_trustnet.csv",sep = ' ')
ciao.columns = ['Column1','Column2']
ciao_trust = cial_trust.sort_values(by=['1'])
freq = cial_trust['1'].value_counts().reset_index()
trustor = freq[freq['count']>1]['1'].tolist()
trust_df = cial_trust[cial_trust['1'].isin(trustor)]
trust_df.to_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/processed_ciao_trustnet.csv"")

epinions_trust = pd.read_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/epinions/epinions_trustnet.csv",sep = ' ')
epinions_trust = epinions_trust.sort_values(by=['2'])
freq = epinions_trust['2'].value_counts().reset_index()
trustor = freq[freq['count']>1]['2'].tolist()
trust_df = epinions_trust[epinions_trust['2'].isin(trustor)]
trust_df.to_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/epinions/processed_epinions_trustnet.csv"")
'''

#rating dataset
ciao_rating = pd.read_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/ciao_rating.csv",sep = ',')
product_rating = ciao_rating['productid'].value_counts().reset_index()
processed_ciao_rating = product_rating[product_rating['count'] > 1]
ciao_rating[ciao_rating['productid'].isin(processed_ciao_rating['count'].tolist())].to_csv("ciao/processed_ciao_rating.csv", index = False)

#ciao_rating['productid'].nunique(),processed_ciao_rating['productid'].nunique()

epinions_rating = pd.read_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/epinions/epinions_rating.csv",sep = ',')
product_rating = epinions_rating['productid'].value_counts().reset_index()
processed_epinions_rating = product_rating[product_rating['count'] > 1]
epinions_rating[epinions_rating['productid'].isin(processed_epinions_rating['count'].tolist())].to_csv("epinions/processed_epinions_rating.csv", index = False)
#epinions_rating['productid'].nunique(),processed_epinions_rating['productid'].nunique()



#distribution of trustor and trustee
column_names = ['Trustor', 'Trustee']
epinions_trust = pd.read_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/epinions/renumbered_graph_epinions.csv",sep = ' ',names=column_names)
freq = epinions_trust['Trustor'].value_counts().reset_index()
epinions_trust = pd.read_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/renumbered_graph_ciao.csv",sep = ' ',names=column_names)
freq = epinions_trust['Trustor'].value_counts().reset_index()




column_names = ['Trustor', 'Trustee']
trustnet = pd.read_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/renumbered_graph_ciao.csv",sep = ' ',, names=column_names)



print('Clustering coefficient: ', nx.average_clustering(graph))
density = 2 * graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1))
print(density)

degrees = dict(G.degree())

# Calculate the average degree
average_degree = sum(degrees.values()) / len(degrees)

print("Degrees of vertices:", degrees)
print("Average degree:", average_degree)



# Extract trustor and trustee data
trustor_data = [edge[0] for edge in G.edges()]
trustee_data = [edge[1] for edge in G.edges()]

# Count occurrences
trustor_counts = Counter(trustor_data)
trustee_counts = Counter(trustee_data)

# Plot the distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.bar(trustor_counts.keys(), trustor_counts.values(), color='blue', alpha=0.7)
ax1.set_title('Trustor Distribution')
ax1.set_xlabel('Trustor Nodes')
ax1.set_ylabel('Count')

ax2.bar(trustee_counts.keys(), trustee_counts.values(), color='green', alpha=0.7)
ax2.set_title('Trustee Distribution')
ax2.set_xlabel('Trustee Nodes')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.show()
