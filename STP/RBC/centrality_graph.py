import numpy as np
import networkx as nx
import fun_network as net
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from statistics import mean
###############################################################################
# Setup paths and filenames
###############################################################################
psi = np.genfromtxt('00X_MX.csv', delimiter=',')
bq_xy = np.genfromtxt('00X_XY.csv', delimiter=',')
##############################################################################
# Transitions Matrix and Base Netowrk
##############################################################################
np.fill_diagonal(psi, 0)
psiN = normalize(psi, axis=1, norm='l2')
G = nx.from_numpy_matrix(psiN)
G.remove_edges_from(nx.selfloop_edges(G))
G = net.calcNetworkDistance(G)
##############################################################################
# Centrality Detection
##############################################################################
centrality_nodes = nx.load_centrality(G, weight='distance')
centrality_edges = nx.edge_betweenness_centrality(G, weight='distance')
##############################################################################
# Export
##############################################################################
final_G = nx.DiGraph()

for i in range(len(bq_xy)):
    final_G.add_node(i, pos=(bq_xy[i][0], bq_xy[i][1]))

for item in centrality_edges.items():
    edge = item[0]
    centrality = item[1]
    if centrality > .01:
        final_G.add_edge(int(edge[0]), int(edge[1]), weight=centrality)

pos = nx.get_node_attributes(final_G, 'pos')
node_sizes = [x * 100 for x in list(centrality_nodes.values())]
nodelist = final_G.nodes()

widths = nx.get_edge_attributes(final_G, 'weight')
edgelist = widths.keys()
edge_sizes = [x * 10 for x in list(widths.values())]

plt.figure(figsize=(12,8))

nx.draw_networkx_nodes(final_G,pos,
                       nodelist=nodelist,
                       node_size=node_sizes,
                       node_color='black',
                       alpha=0.7)
nx.draw_networkx_edges(final_G,pos,
                       edgelist=edgelist,
                       width=edge_sizes,
                       edge_color='lightblue',
                       alpha=.6)

plt.savefig("mosquito_centrality_20.png")
