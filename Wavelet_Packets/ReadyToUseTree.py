import numpy as np
import matplotlib.pyplot as plt
import torch 
from .QuadTree import Tree

# Usefull Tree
def Tree_Depth(depth,draw=False):
    tree = Tree()

    if depth > 1 :
        Nodes_list=[['0']]
        for n in range(1,depth):
            New_Nodes =[]
            for node in Nodes_list[-1]:
                New_Nodes.append(node+str(0))
                New_Nodes.append(node + str(1))
                New_Nodes.append(node + str(2))
                New_Nodes.append(node + str(3))
            Nodes_list.append(New_Nodes)
        for n in range(1, depth):
            for node in Nodes_list[n]:
                tree.addNode(node)
    if draw == True:
        tree.draw()
    return(tree)

def Tree_Depth_High(depth,draw=False):
    tree = Tree()

    if depth > 1 :
        Nodes_list=[['0']]
        for n in range(1,depth):
            New_Nodes =[]
            for node in Nodes_list[-1]:
                if n>1:
                  New_Nodes.append(node+str(0))
                New_Nodes.append(node + str(1))
                New_Nodes.append(node + str(2))
                New_Nodes.append(node + str(3))
            Nodes_list.append(New_Nodes)
        for n in range(1, depth):
            for node in Nodes_list[n]:
                tree.addNode(node)
    if draw == True:
        tree.draw()
    return(tree)
    
def Tree_Wav(depth,draw=False):
    tree = Tree()

    if depth > 1 :
        Nodes_list=[['0']]
        for n in range(1,depth):
            New_Nodes =[]
            for node in Nodes_list[-1]:
                New_Nodes.append(node+str(0))
            Nodes_list.append(New_Nodes)
        for n in range(1, depth):
            for node in Nodes_list[n]:
                tree.addNode(node)
    if draw == True:
        tree.draw()
    return(tree)
    
def Tree_WavOctaves(depth,Q,draw=False):
    tree = Tree()

    for index in range(1,depth): 
        if index>1:
          tree.addNode('0'*index)
          
        Nodes_list=[['0'*index]]
        for n in range(1,Q):
            New_Nodes =[]
            for node in Nodes_list[-1]:
                if n>1:
                  New_Nodes.append(node+str(0))
                New_Nodes.append(node + str(1))
                New_Nodes.append(node + str(2))
                New_Nodes.append(node + str(3))
            Nodes_list.append(New_Nodes)
        for n in range(1, Q):
            for node in Nodes_list[n]:
                tree.addNode(node)
    if draw == True:
        tree.draw()
    return(tree)