
import numpy as np


def calcNetworkDistance(G):
    nodesNum = len(G)
    for i in range(nodesNum):
        keys = G[i]
        for j in range(nodesNum):
            prb = keys.get(j)
            if prb is not None:
                weight = prb['weight']
                if weight > 0:
                    distance = 1 / prb['weight']
                else:
                    distance = np.Inf
                G[i][j]['distance'] = distance
    return G