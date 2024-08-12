#  <#Title#>

## Matlab Code
load('YelpHotel.mat')
csvwrite('YelpHotel.csv',Label)
G = graph(Network,'upper')

edgesTable = G.Edges;
filename = 'graph_edges.csv';
writetable(edgesTable, filename);
 ###

df = pd.read_csv("YelpHotel/graph_edges.csv")

df = df[df['Weight'] > 0]

df.to_csv("YelpHotel_graph.csv")

df_nodes = pd.read_csv('YelpHotel/YelpHotel.csv',header = None)
df_nodes = df_nodes.reset_index()
# df_reset = df_nodes.reset_index()

# Set the index to start from 1
df_nodes['index'] = df_nodes['index'] + 1

df_nodes = df_nodes.rename(columns={'index': 'Node',0:'Anomaly'})
df_nodes.to_csv("YelpHotel_labels.csv",index = False)
