import matplotlib.pyplot as plt
import torch 
import numpy as np


"""QUADTREE USED BY WAVELET PACKETS"""
class TreeNode:
    def __init__(self, depth, name, index_x, index_y, gray_x, gray_y):
        self.aa = None
        self.ad = None
        self.da = None
        self.dd = None
        self.depth = depth
        self.name = name
        self.index_x = index_x
        self.index_y = index_y
        self.gray_x = gray_x
        self.gray_y = gray_y


class Tree:
    def __init__(self):
        self.root = None
        # self.nodes_list = {'00':'00','01':'01','02':'02','03':'03'}
        self.nodes_list = {'00': (0, 0.5, 0.5, 1), '01': (0.5, 1, 0.5, 1), '02': (0, 0.5, 0, 0.5),
                           '03': (0.5, 1, 0, 0.5)}
        self.reconstruct = ['0']
        self.nodes = TreeNode(1, '0', 0, 0, 0, 0)

    def addNode(self, key):
        # add Node

        # self.nodes_list.pop(key)
        # self.nodes_list[key+'0']=key+'0'
        # self.nodes_list[key+'1']=key+'1'
        # self.nodes_list[key+'2']=key+'2'
        # self.nodes_list[key+'3']=key+'3'
        x1, x2, y1, y2 = self.nodes_list.pop(key)

        self.nodes_list[key + '0'] = (x1, (x2 + x1) / 2, (y1 + y2) / 2, y2)
        self.nodes_list[key + '1'] = ((x1 + x2) / 2, x2, (y1 + y2) / 2, y2)
        self.nodes_list[key + '2'] = (x1, (x2 + x1) / 2, y1, (y1 + y2) / 2)
        self.nodes_list[key + '3'] = ((x1 + x2) / 2, x2, y1, (y1 + y2) / 2)

        # reconstruction order
        self.reconstruct = [key] + self.reconstruct

        self.updateNode(key)

    def draw(self):
        for rect in self.nodes_list.items():
            x1, x2, y1, y2 = rect[1]

            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c='black')
        plt.show()

    def updateGray(self, G_p, mode):
        if mode == '2p':
            if G_p % 2 == 0:
                return 2 * G_p
            else:
                return 2 * G_p + 1

        if mode == '2p+1':
            if G_p % 2 == 0:
                return 2 * G_p + 1
            else:
                return 2 * G_p

    def updateNode(self, key):
        # Key is the frequency ordering index
        node = self.nodes
        for j in range(1, len(key) - 1):
            if key[j] == '0':
                node = node.aa
            elif key[j] == '1':
                node = node.ad
            elif key[j] == '2':
                node = node.da
            elif key[j] == '3':
                node = node.dd

        # We have to be carefull, even tho we label as aa, ad and da the corners, this is just frequency ordering, and the filters applied might not be aa ==hh and dd = gg

        index_x, index_y, gray_x, gray_y = node.index_x, node.index_y, node.gray_x, node.gray_y

        if index_x % 2 == 0:
            if index_y % 2 == 0:
                if key[-1] == '0':
                    node.aa = TreeNode(len(key), key, 2 * index_x, 2 * index_y, self.updateGray(node.gray_x, '2p'),
                                       self.updateGray(node.gray_y, '2p'))
                elif key[-1] == '1':
                    node.ad = TreeNode(len(key), key, 2 * index_x, 2 * index_y + 1, self.updateGray(node.gray_x, '2p'),
                                       self.updateGray(node.gray_y, '2p'))
                elif key[-1] == '2':
                    node.da = TreeNode(len(key), key, 2 * index_x + 1, 2 * index_y,
                                       self.updateGray(node.gray_x, '2p'), self.updateGray(node.gray_y, '2p'))
                elif key[-1] == '3':
                    node.dd = TreeNode(len(key), key, 2 * index_x + 1, 2 * index_y + 1,
                                       self.updateGray(node.gray_x, '2p'), self.updateGray(node.gray_y, '2p'))

            else:
                if key[-1] == '0':
                    node.aa = TreeNode(len(key), key, 2 * index_x, 2 * index_y + 1, self.updateGray(node.gray_x, '2p'),
                                       self.updateGray(node.gray_y, '2p+1'))
                elif key[-1] == '1':
                    node.ad = TreeNode(len(key), key, 2 * index_x, 2 * index_y, self.updateGray(node.gray_x, '2p'),
                                       self.updateGray(node.gray_y, '2p+1'))
                elif key[-1] == '2':
                    node.da = TreeNode(len(key), key, 2 * index_x + 1, 2 * index_y + 1,
                                       self.updateGray(node.gray_x, '2p'), self.updateGray(node.gray_y, '2p+1'))
                elif key[-1] == '3':
                    node.dd = TreeNode(len(key), key, 2 * index_x + 1, 2 * index_y,
                                       self.updateGray(node.gray_x, '2p'), self.updateGray(node.gray_y, '2p+1'))

        else:

            if index_y % 2 == 0:
                if key[-1] == '0':
                    node.aa = TreeNode(len(key), key, 2 * index_x + 1, 2 * index_y,
                                       self.updateGray(node.gray_x, '2p+1'), self.updateGray(node.gray_y, '2p'))
                elif key[-1] == '1':
                    node.ad = TreeNode(len(key), key, 2 * index_x + 1, 2 * index_y + 1,
                                       self.updateGray(node.gray_x, '2p+1'), self.updateGray(node.gray_y, '2p'))
                elif key[-1] == '2':
                    node.da = TreeNode(len(key), key, 2 * index_x, 2 * index_y, self.updateGray(node.gray_x, '2p'),
                                       self.updateGray(node.gray_y, '2p+1'))
                elif key[-1] == '3':
                    node.dd = TreeNode(len(key), key, 2 * index_x, 2 * index_y + 1, self.updateGray(node.gray_x, '2p+1'),
                                       self.updateGray(node.gray_y, '2p'))

            else:
                if key[-1] == '0':
                    node.aa = TreeNode(len(key), key, 2 * index_x + 1, 2 * index_y + 1,
                                       self.updateGray(node.gray_x, '2p+1'), self.updateGray(node.gray_y, '2p+1'))
                elif key[-1] == '1':
                    node.ad = TreeNode(len(key), key, 2 * index_x + 1, 2 * index_y,
                                       self.updateGray(node.gray_x, '2p+1'), self.updateGray(node.gray_y, '2p+1'))
                elif key[-1] == '2':
                    node.da = TreeNode(len(key), key, 2 * index_x, 2 * index_y + 1, self.updateGray(node.gray_x, '2p+1'),
                                       self.updateGray(node.gray_y, '2p+1'))
                elif key[-1] == '3':
                    node.dd = TreeNode(len(key), key, 2 * index_x, 2 * index_y, self.updateGray(node.gray_x, '2p+1'),
                                       self.updateGray(node.gray_y, '2p+1'))

