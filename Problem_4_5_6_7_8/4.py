# Write a simple program to do: A dataset collected in a cosmetics shop showing
# details of customers and whether or not they responded to a special offer
# to buy a new lip-stick is shown in table below. (Implement step by step
# using commands - Dont use library). Use this dataset to build a decision
# tree, with Buys as the target variable, to help in buying lipsticks in the
# future. Find the root node of the decision tree.

import csv
import math
from collections import Counter, defaultdict

TARGET = "Buys"   # name of target column in CSV

def load_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
    return data, reader.fieldnames

def entropy(rows):
    # compute entropy for target column
    counts = Counter(row[TARGET] for row in rows)
    total = len(rows)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent

def partition_by_attribute(rows, attr):
    # returns dict value -> list of rows
    parts = defaultdict(list)
    for r in rows:
        parts[r[attr]].append(r)
    return parts

def info_gain(rows, attr):
    base_entropy = entropy(rows)
    parts = partition_by_attribute(rows, attr)
    total = len(rows)
    weighted_ent = 0.0
    for subset in parts.values():
        weighted_ent += (len(subset) / total) * entropy(subset)
    return base_entropy - weighted_ent

def majority_value(rows):
    counts = Counter(row[TARGET] for row in rows)
    if not counts:
        return None
    return counts.most_common(1)[0][0]

class Node:
    def __init__(self, attribute=None, is_leaf=False, prediction=None):
        self.attribute = attribute
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.children = {}   # value -> Node

    def __repr__(self):
        if self.is_leaf:
            return "Leaf({})".format(self.prediction)
        else:
            return "Node({})".format(self.attribute)

def id3(rows, attributes):
    # If no examples, return leaf with None (or could use default)
    if not rows:
        return Node(is_leaf=True, prediction=None)

    # If all examples have same target, return leaf with that target
    targets = [r[TARGET] for r in rows]
    if len(set(targets)) == 1:
        return Node(is_leaf=True, prediction=targets[0])

    # If no attributes left, return majority
    if not attributes:
        return Node(is_leaf=True, prediction=majority_value(rows))

    # Choose best attribute
    gains = [(attr, info_gain(rows, attr)) for attr in attributes]
    best_attr, best_gain = max(gains, key=lambda x: x[1])

    root = Node(attribute=best_attr)
    # Partition and recurse
    parts = partition_by_attribute(rows, best_attr)
    remaining_attrs = [a for a in attributes if a != best_attr]
    for attr_val, subset in parts.items():
        child = id3(subset, remaining_attrs)
        root.children[attr_val] = child
    return root

def print_tree(node, indent=""):
    if node.is_leaf:
        print(indent + "=> Predict:", node.prediction)
    else:
        if indent == "":
            print(indent + "[Root attribute: " + node.attribute + "]")
        else:
            print(indent + "Attribute: " + node.attribute)
        for val, child in node.children.items():
            print(indent + "  If {} = {}:".format(node.attribute, val))
            print_tree(child, indent + "    ")

def find_root_attribute(path_to_csv):
    data, headers = load_csv(path_to_csv)
    if TARGET not in headers:
        raise ValueError("Target column '{}' not found in CSV headers: {}".format(TARGET, headers))
    attributes = [h for h in headers if h != TARGET]
    # Compute info gains
    gains = []
    for a in attributes:
        g = info_gain(data, a)
        gains.append((a, g))
    gains.sort(key=lambda x: x[1], reverse=True)
    return gains, data, attributes

def main():
    csv_path = "Cosmetics_Shop.csv"
    # csv_path = "check.csv"
    gains, data, attributes = find_root_attribute(csv_path)
    print("Information gain for each attribute (sorted):")
    for a, g in gains:
        print("  {:20s} : {:.6f}".format(a, g))

    if gains:
        root_attr, root_gain = gains[0]
        print("\nRoot node (attribute with highest information gain):", root_attr)
    else:
        print("No attributes found in CSV")

    # Build full tree (optional)
    tree = id3(data, attributes)
    print("\nDecision tree (readable):")
    print_tree(tree)

if __name__ == "__main__":
    main()
