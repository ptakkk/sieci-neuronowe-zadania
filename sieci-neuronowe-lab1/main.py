import numpy as np
import csv
from collections import Counter

VARIABLE_NUMBER = [1, 3, 4, 5, 6, 7]   # Pclass, Sex, Age, SibSp, Parch, Survived


class Node:
    def __init__(self):
        self.parent_variable_value = None
        self.feature = None
        self.decision = None
        self.children = []

    def add_child(self, node):
        self.children.append(node)


def read_file():
    rows = []
    with open("titanic-homework.csv", newline="") as file:
        reader = csv.reader(file)
        
        header = next(reader)  
        rows.append(header)
        
        for row in reader:
            age = float(row[4])
            if age < 20:
                row[4] = 'young'
            elif age < 40:
                row[4] = 'middle'
            else:
                row[4] = 'old'
            rows.append(row)

    data = np.array(rows)
    variable_names = data[0, VARIABLE_NUMBER].copy()
    data = data[1:, VARIABLE_NUMBER]

    return data, variable_names


def calc_entropy(x):
    counts = Counter(x)
    probabilities = [count / len(x) for count in counts.values()]
    entropy_value = -sum(p * np.log2(p) for p in probabilities)

    return entropy_value


def calc_gain_ratio(data, feature, decision_entropy):
    entropy_sum = 0

    variable_values = np.unique(data[:, feature])
    for value in variable_values:
        data_part = data[data[:, feature] == value]
        data_part_decision = data_part[:, -1]

        entropy_sum += data_part.shape[0] / data.shape[0] * calc_entropy(data_part_decision)

    information_gain = decision_entropy - entropy_sum
    split = calc_entropy(data[:, feature])
    
    if split == 0:
        gain_ratio = float('inf') 
    else:
        gain_ratio = information_gain / split
    return gain_ratio


def decision_tree(node: Node, data, variables):
    if data.size == 0 or len(variables) == 0:
        node.decision = "Undecided"
        return

    if np.all(data[:, -1] == "0"):
        node.decision = "Died"
        return

    if np.all(data[:, -1] == "1"):
        node.decision = "Survived"
        return

    decision_entropy = calc_entropy(data[:, -1])

    best_feature = None
    best_gain_ratio = -1

    for feature in variables:
        gain_ratio = calc_gain_ratio(data, feature, decision_entropy)
        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_feature = feature

    node.feature = best_feature

    variable_values = np.unique(data[:, best_feature])
    for value in variable_values:
        child = Node()
        child.parent_variable_value = value
        node.add_child(child)
        new_data = data[data[:, best_feature] == value]
        remaining_variables = list(filter(lambda x: x != best_feature, variables))
        decision_tree(child, new_data, remaining_variables)


def draw_tree(root: Node, variable_names):
    def draw_node(node: Node, indent=""):
        if node.feature is not None:
            print(indent + "Feature:", variable_names[node.feature])
        else:
            label = node.decision
            print(indent + "Decision:", label)

        for child in node.children:
            print(indent + "Edge:", child.parent_variable_value)
            draw_node(child, indent + "\t")
    draw_node(root)


if __name__ == "__main__":
    data, variable_names = read_file()
    tree = Node()
    decision_tree(tree, data, range(5))
    draw_tree(tree, variable_names)