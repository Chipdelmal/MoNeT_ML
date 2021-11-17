from networkx.readwrite import nx_shp
import numpy as np
import networkx as nx
import fun_network as net
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from statistics import mean
import math
###############################################################################
# Setup paths and filenames
###############################################################################
psi = np.genfromtxt('LRG_01-350-HOM_MX.csv', delimiter=',')
bq_xy = np.genfromtxt('LRG_01-350-HOM_XY.csv', delimiter=',')
shape_options = ['s', 'o', 'd']
color_options = ['#e0c3fc', '#caffbf', '#a0c4ff']
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
    final_G.add_node(i, pos=(bq_xy[i][0], bq_xy[i][1]), 
        shape=shape_options[int(bq_xy[i][2])], 
        color=color_options[int(bq_xy[i][2])], 
        size=list(centrality_nodes.values())[i])

for item in centrality_edges.items():
    edge = item[0]
    centrality = item[1]
    final_G.add_edge(int(edge[0]), int(edge[1]), 
        weight=centrality)

pos = nx.get_node_attributes(final_G, 'pos')

widths = nx.get_edge_attributes(final_G, 'weight')
edge_sizes = set(list(widths.values()))

plt.figure(figsize=(12,8))

# na & ea corresponds to transparency: alpha > 1: less transparent, alpha < 1: more transparent
# ns & ew corresponds to size/width
default_alpha = 1
ns=10000000000
na=50
ew=10
ea=100

for shape in set(shape_options):
    node_list = [node for node in final_G.nodes() if final_G.nodes[node]['shape']==shape]
    nx.draw_networkx_nodes(final_G, pos,
                        nodelist=node_list,
                        node_size=[math.log(1+ns*final_G.nodes[node]['size']) for node in node_list],
                        node_color=[final_G.nodes[node]['color'] for node in node_list],
                        node_shape=shape,
                        alpha=[min(default_alpha, math.log(1+na*final_G.nodes[node]['size'])) for node in node_list])

for width in edge_sizes:
    edge_list = [edge for edge in final_G.edges() if final_G.edges[edge]['weight']==width]
    nx.draw_networkx_edges(final_G,pos,
                        edgelist=edge_list,
                        width=math.log(1+ew*width),
                        edge_color='black',
                        alpha=min(default_alpha, math.log(1+ea*width)))

plt.savefig("350-HOM_centrality2.png", bbox_inches='tight', pad_inches=0)
