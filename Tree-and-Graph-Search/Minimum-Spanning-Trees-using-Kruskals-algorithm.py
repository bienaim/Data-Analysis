import networkx as nx
import heapq as pq

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 19:09:04 2018

@author: abien
"""

#Precondition: Graph G is connected, undirected, and weighted
#PostCondition: Forest F is the Minimum Spanning Tree of graph G
#
# The goals for the Kruskal algorithm are as follows:
#CGa (Parts) Forest F consists of trees, which are parts of a Minimum Spanning Tree for graph G
#[CGb] (Greed Used) All changes to F add the lowest weighted edge that doesn’t create a cycle
#CGc (Complement) F is a minimum spanning tree for graph G

EdgeList = []
F = nx.Graph()
moreEdges = 'y'

#User input for vertices and edges
while moreEdges != 'n':
    myEdges = input("Please enter an edge as two vertices (letters) and a weight (an integer from 1 to 9),"
                    +"\n separating each with a comma e.g. A,B,2 \n")
    EdgeList.append(tuple(myEdges.split(',')))
    moreEdges = input("Are there more edges to add? Please enter 'y' or 'n'.\n")
print('\nGraph G contains' + str(EdgeList) + "\n")

heap = []
i = 0
edgeVertex = []
#CGa (Parts) Forest F consists of trees, which are parts of a Minimum Spanning Tree for graph G
while i < len(EdgeList):
     
     #Vertices are added to forest
     F.add_node(EdgeList[i][0])
     F.add_node(EdgeList[i][1])
     #Edges are added to edgeVertex list
     edgeVertex.append((int(EdgeList[i][2]),EdgeList[i][0],EdgeList[i][1]))
     i += 1

#Edges and weights are added to heap from edgeVertex List 
pq.heapify(edgeVertex)   
i = 0    
while i < len(edgeVertex):
    pq.heappush(heap, edgeVertex[i])
    i += 1
heap.sort()

print('Forest F contains vertices ' + str(F.nodes) + '\n')

print('The heap contains edges ' + str(heap) + "\n")

#[CGb] (Greed Used) All changes to F add the lowest weighted edge that doesn’t create a cycle
tree = 'False'
while tree == 'False':
    #Remove an edge from the heap
    minWeightEdge = pq.heappop(heap)
    verticesAndWeightedEdges = str(minWeightEdge[1]) + ',' + str(minWeightEdge[2]) + ',' + str(minWeightEdge[0])
    weightedEdgeForGraph = []
    weightedEdgeForGraph.append(tuple(verticesAndWeightedEdges.split(',')))
    #Add the edge to the set
    F.add_weighted_edges_from(weightedEdgeForGraph)
    
    #If the added edge creates a cycle, remove it from the minimum spanning tree
    if nx.cycle_basis(F,minWeightEdge[1]) != []:
        print("Adding weighted edge " + str(minWeightEdge[1]) + " " + str(minWeightEdge[2]) + " results in a cycle.\n")        
        F.remove_edge(str(minWeightEdge[1]),str(minWeightEdge[2]))
        tree = str(nx.algorithms.tree.recognition.is_tree(F))
        
    else:
        print("Weighted edge " + str(minWeightEdge[1]) + " " + str(minWeightEdge[2]) + " has been removed from the heap and added to F.")
        tree = str(nx.algorithms.tree.recognition.is_tree(F))
        if tree == 'False':
            print("F is a forest.\n")
        
#CGc (Complement) F is a minimum spanning tree for graph G
print('F is a minimum spanning tree for Graph G containing edges:')
print(F.edges)
options = {"node_size": 1000, "alpha": 0.4}
nx.draw(F, with_labels = True, font_weight = 'bold', font_size=20, node_color="r", edge_color="g", width=6,  **options)
