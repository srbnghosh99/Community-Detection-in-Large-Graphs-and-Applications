import simplejson as json
import networkx as nx
import shlex
import sys
import igraph as ig


g = nx.karate_club_graph()
# fig, ax = plt.subplots(1, 1, figsize=(8, 6));
# nx.draw_networkx(g, ax=ax)
nodes = [{'name': str(i), 'club': g.nodes[i]['club']}
         for i in g.nodes()]
links = [{'source': u[0], 'target': u[1]}
         for u in g.edges()]
with open('graph.json', 'w') as f:
    json.dump({'nodes': nodes, 'links': links},
              f, indent=4,)
