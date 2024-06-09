import numpy as np
import math
import csv

def read_data(filename):
    with open(filename, 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        headers = next(datareader)
        metadata = headers
        traindata = [row for row in datareader]

    return metadata, traindata

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""
        
    def __str__(self):
        return self.attribute

def subtables(data, col, delete):
    unique_values, subtable_dict = np.unique(data[:, col], return_inverse=True), {}
    for value in unique_values:
        subtable_dict[value] = data[data[:, col] == value]
        if delete:
            subtable_dict[value] = np.delete(subtable_dict[value], col, 1)
    return unique_values, subtable_dict

def entropy(S):
    unique_values, counts = np.unique(S, return_counts=True)
    probabilities = counts / S.size
    return -np.sum(probabilities * np.log2(probabilities))

def gain_ratio(data, col):
    unique_values, subtable_dict = subtables(data, col, delete=False)
    total_size = data.shape[0]
    entropies = np.array([subtable_dict[value].shape[0] / total_size * entropy(subtable_dict[value][:, -1]) for value in unique_values])
    total_entropy = entropy(data[:, -1])
    iv = -np.sum((subtable_dict[value].shape[0] / total_size) * np.log2(subtable_dict[value].shape[0] / total_size) for value in unique_values)
    return (total_entropy - np.sum(entropies)) / iv

def create_node(data, metadata):
    if np.unique(data[:, -1]).shape[0] == 1:
        node = Node("")
        node.answer = np.unique(data[:, -1])[0]
        return node
        
    gains = np.array([gain_ratio(data, col) for col in range(data.shape[1] - 1)])
    split = np.argmax(gains)
    
    node = Node(metadata[split])    
    metadata = np.delete(metadata, split, 0)    
    
    unique_values, subtable_dict = subtables(data, split, delete=True)
    
    for value in unique_values:
        child = create_node(subtable_dict[value], metadata)
        node.children.append((value, child))
    
    return node

def print_tree(node, level=0):
    if node.answer != "":
        print("   " * level, node.answer)
        return
    print("   " * level, node.attribute)
    for value, n in node.children:
        print("   " * (level + 1), value)
        print_tree(n, level + 2)

metadata, traindata = read_data("tennisdata.csv")
data = np.array(traindata)
node = create_node(data, metadata)
print_tree(node)
