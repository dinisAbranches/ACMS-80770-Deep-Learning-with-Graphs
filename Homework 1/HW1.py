"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 1: Programming assignment
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import bipartite
from networkx.generators.random_graphs import erdos_renyi_graph
import copy


# -- Initialize graphs
seed = 30
G = nx.florentine_families_graph()
nodes = G.nodes()

layout = nx.spring_layout(G, seed=seed)

# -- compute jaccard's similarity
"""
    This example is using NetwrokX's native implementation to compute similarities.
    Write a code to compute Jaccard's similarity and replace with this function.
"""

# -- Function to compute the Jaccard's similarity between two nodes
def computeJaccardSimilarity(G,v1,v2):
    """
    computeJaccardSimilarity() computes the Jaccard's Similarity between nodes
    v1 and v2 in graph G.

    Parameters
    ----------
    G : networkx.classes.graph.Graph object
        Graph object of interest.
    v1 : networkx node hashable object
        First node.
    v2 : networkx node hashable object
        Second node.

    Returns
    -------
    S_12 : float
        Jaccard's Similarity between nodes v1 and v2 in graph G.

    """
    # Get neighbors of node v1
    N_v1=list(G.neighbors(v1))
    # Get neighbors of node v2
    N_v2=list(G.neighbors(v2))
    # Get intersection list
    intersect=list(set(N_v1)&set(N_v2))
    # Get union list
    union=list(set(N_v1)|set(N_v2))
    # Compute Jaccard's Similarity
    S_12=len(intersect)/len(union)
    # Check against native implementation (can be commented out)
    if S_12!=list(nx.jaccard_coefficient(G,[(v1,v2)]))[0][2]:
        raise ValueError('Inconcistency in the S_12 value of the node pair:\n'
                         +'( '+str(v1)+' , '+str(v2)+' )')
    # Output
    return S_12

# -- Function to compute the Jaccard's matrix of G
def computeJaccardMatrix(G,algorithm='1'):
    """
    computeJaccardMatrix() computes the Jaccard Similarity Matrix of G.

    Parameters
    ----------
    G : networkx.classes.graph.Graph object
        Graph object of interest.
    algorithm : sting, optional
        Algorithm to be used. One of:
            '1' - Vectorized algorithm (faster)
            '2' - Non-vectorized (double loop) algorithm (useful to validate
                                                          against native
                                                          implementation)
        The default is '1'.

    Returns
    -------
    S : numpy array
        Jaccard Similarity matrix.

    """
    # Check algorithm
    if algorithm=='1':
        # Get adjacency matrix of G
        A=nx.to_numpy_array(G)
        # Get matrix of number of common neighbors
        intersectMatrix=np.matmul(A.T,A)
        # Get union matrix (total number of unique neighbors between i and j)
        unionMatrix=A.sum(axis=0).reshape(-1,1).repeat(len(A),axis=1) \
                    +A.sum(axis=1).reshape(1,-1).repeat(len(A),axis=0) \
                    -intersectMatrix
        # Compute Jaccard's Matrix
        S=np.divide(intersectMatrix,unionMatrix)
    elif algorithm=='2':
        # Get list of nodes
        nodeList=list(G.nodes)
        # Get total number of nodes
        N=len(nodeList)
        # Intialize Jaccard's Matrix
        S=np.zeros([N,N])
        # Iterate over nodes (double loop; inefficient)
        for i in range(N):
            for j in range(N):
                # Compute Jaccard's Similarity between nodes i and j
                S_12=computeJaccardSimilarity(G,nodeList[i],nodeList[j])
                # Update S
                S[i,j]=S_12
    # Output
    return S

# -- Compute similarities between the Ginori node and all other nodes
# Validate both algorithms in computeJaccardMatrix() (can be commented out)
if not np.all(computeJaccardMatrix(G,algorithm='1')
              ==computeJaccardMatrix(G,algorithm='2')):
    raise ValueError('Algorithm inconsistency in computeJaccardMatrix()')
# Compute Jaccard's Matrix
S=computeJaccardMatrix(G)
# Define central node of interest
node='Ginori'
# Get index of node "Ginori"
index=list(nodes).index(node)
# Get Ginori-based similarities
S_Ginori=S[index,:]
# Get list of Ginori-based artificial (or not) edges
new_edges=[(node,v2) for v2 in nodes]
# Remove self edge (Ginori,Ginori)
S_Ginori=np.delete(S_Ginori,index)
new_edges.pop(index)
# Print similarities
for metric,edge in zip(S_Ginori,new_edges):
    print(f"({edge[0]}, {edge[1]}) -> {metric:.8f}")

# -- plot Florentine Families graph
nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_size=600)
nx.draw_networkx_edges(G, edgelist=G.edges(), pos=layout, edge_color='gray', width=4)

# -- plot edges representing similarity
nx.draw_networkx_labels(G,pos=layout,font_size=10,font_weight='bold')
ne=nx.draw_networkx_edges(G,edgelist=new_edges,pos=layout,
                          edge_color=np.asarray(S_Ginori),width=4,alpha=0.7)
plt.colorbar(ne)
plt.axis('off')
plt.show()

# -- plot edges representing non-zero similarity
# Remove all metrics equal to zero
new_edges_nonzero=[aux2 for aux1,aux2 in zip(S_Ginori!=0,new_edges) if aux1]
S_Ginori_nonzero=S_Ginori[S_Ginori!=0]
# Plot
nx.draw_networkx_nodes(G,nodelist=nodes,label=nodes,pos=layout,node_size=600)
nx.draw_networkx_labels(G,pos=layout,font_size=10,font_weight='bold')
nx.draw_networkx_edges(G, edgelist=G.edges(), pos=layout, edge_color='gray', width=4)
ne=nx.draw_networkx_edges(G,edgelist=new_edges_nonzero,pos=layout,
                          edge_color=np.asarray(S_Ginori_nonzero),
                          width=4,alpha=0.7)
plt.colorbar(ne)
plt.axis('off')
plt.show()