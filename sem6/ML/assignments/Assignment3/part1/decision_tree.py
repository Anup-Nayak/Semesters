"""
COL774 Assignment 3: Decision Trees & Neural Networks

File: decision_tree.py
Author: Anup Lal Nayak
Entry Number: <your entry number>

Description:
This file contains all code for Part I (Decision Trees) of Assignment 3.
It includes loading, preprocessing, model training, evaluation, and analysis
as per parts (a) to (f).

Usage:
Run this file as a script after placing the dataset in the appropriate folders.
"""

# ==== Imports ====
import pandas as pd
import numpy as np
import sys
import os
import math
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# Global variables for data
train_X = None
train_y = None
valid_X = None
valid_y = None
test_X = None   

output_folder_path = None

# ==== Utility Functions ====
def load_data(train_path, valid_path, test_path):
    """Load and preprocess the data."""
    # Load datasets
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    # Clean string columns: strip whitespace
    def clean_strings(df):
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()
        return df

    train_df = clean_strings(train_df)
    valid_df = clean_strings(valid_df)
    test_df = clean_strings(test_df)

    # Rename target column for consistency
    train_df.rename(columns={"income": "label"}, inplace=True)
    valid_df.rename(columns={"income": "label"}, inplace=True)

    # Separate features and labels
    train_X, train_y = train_df.drop(columns=["label"]), train_df["label"]
    valid_X, valid_y = valid_df.drop(columns=["label"]), valid_df["label"]
    test_X = test_df
    
    return train_X, train_y, valid_X, valid_y, test_X

# ==== Decision Tree ====

# ---------- Node definition ----------
class TreeNode:
    def __init__(self, is_leaf=False, prediction=None, split_attr=None, split_value=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.split_attr = split_attr
        self.split_value = split_value  # Used for numeric splits
        self.children = {}  # key → TreeNode

# ---------- Helper functions ----------
def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

def mutual_info(y, attr_col):
    base_entropy = entropy(y)
    values = np.unique(attr_col)
    split_entropy = 0
    for v in values:
        subset_y = y[attr_col == v]
        if len(subset_y) == 0:
            continue
        split_entropy += (len(subset_y) / len(y)) * entropy(subset_y)
    return base_entropy - split_entropy

def best_split(X, y):
    best_gain = -1
    best_attr = None
    best_val = None

    for col in X.columns:
        if X[col].dtype == 'object':
            gain = mutual_info(y, X[col])
            if gain > best_gain:
                best_gain = gain
                best_attr = col
                best_val = None  # categorical attribute
        else:
            median = X[col].median()
            split_col = X[col] > median
            gain = mutual_info(y, split_col)
            if gain > best_gain:
                best_gain = gain
                best_attr = col
                best_val = median  # numeric attribute

    return best_attr, best_val

def majority_class(y):
    return y.value_counts().idxmax() if len(y) > 0 else None

# ---------- Tree builder ----------
def build_tree(X, y, depth=0, max_depth=10):
    if len(set(y)) == 1:
        return TreeNode(is_leaf=True, prediction=y.iloc[0])
    
    if depth == max_depth or len(X.columns) == 0:
        return TreeNode(is_leaf=True, prediction=majority_class(y))

    split_attr, split_val = best_split(X, y)
    if split_attr is None:
        return TreeNode(is_leaf=True, prediction=majority_class(y))

    node = TreeNode(is_leaf=False, split_attr=split_attr, split_value=split_val)

    if split_val is None:
        # Categorical attribute
        for val in X[split_attr].unique():
            idx = X[split_attr] == val
            if idx.sum() == 0:
                continue
            child = build_tree(X[idx].drop(columns=[split_attr]), y[idx], depth + 1, max_depth)
            node.children[val] = child
    else:
        # Numeric attribute: binary split
        left_idx = X[split_attr] <= split_val
        right_idx = X[split_attr] > split_val

        if left_idx.sum() == 0 or right_idx.sum() == 0:
            return TreeNode(is_leaf=True, prediction=majority_class(y))

        node.children['<='] = build_tree(X[left_idx], y[left_idx], depth + 1, max_depth)
        node.children['>'] = build_tree(X[right_idx], y[right_idx], depth + 1, max_depth)

    return node

# ---------- Prediction ----------
def predict_single(x, node):
    while not node.is_leaf:
        val = x[node.split_attr]
        if node.split_value is None:
            # Categorical
            if val in node.children:
                node = node.children[val]
            else:
                return node.prediction  # Unknown category
        else:
            # Numeric
            key = '>' if val > node.split_value else '<='
            node = node.children[key]
    return node.prediction

def predict(X, tree):
    return X.apply(lambda x: predict_single(x, tree), axis=1)

# ==== Part A: Load and Preprocess ====
def part_a():
    """Load data and preprocess it."""
    tree = build_tree(train_X, train_y, max_depth=20)
    test_preds = predict(test_X, tree)
    
    #save prediction as csv
    output_path = os.path.join(output_folder_path, "prediction_a.csv")
    pd.DataFrame({'prediction':test_preds}).to_csv(output_path,index=False) 
    
# ==== Part B: Decision Tree Training ====
def part_b():
    """Train a decision tree and evaluate."""
    
    # Combine all datasets temporarily for consistent encoding
    combined = pd.concat([train_X, valid_X, test_X], keys=['train', 'valid', 'test'])

    # One-hot encode all categorical columns (with >2 categories)
    combined_encoded = pd.get_dummies(combined)

    # Split back into individual sets
    train_X_encoded = combined_encoded.xs('train')
    valid_X_encoded = combined_encoded.xs('valid')
    test_X_encoded = combined_encoded.xs('test')
    
    tree = build_tree(train_X_encoded, train_y, max_depth=55)
    test_preds = predict(test_X_encoded, tree)
    
    #save prediction as csv
    output_path = os.path.join(output_folder_path, "prediction_b.csv")
    pd.DataFrame({'prediction':test_preds}).to_csv(output_path,index=False) 
    
# ==== Part C: Varying Depth ====
def part_c():
    """Train decision trees of different depths and evaluate."""
    
    # Combine all datasets temporarily for consistent encoding
    combined = pd.concat([train_X, valid_X, test_X], keys=['train', 'valid', 'test'])

    # One-hot encode all categorical columns (with >2 categories)
    combined_encoded = pd.get_dummies(combined)

    # Split back into individual sets
    train_X_encoded = combined_encoded.xs('train')
    valid_X_encoded = combined_encoded.xs('valid')
    test_X_encoded = combined_encoded.xs('test')
    
    def is_leaf(node):
        return node.is_leaf

    def get_internal_nodes(node, path=()):
        """Recursively collect all internal (non-leaf) nodes in the tree."""
        if node.is_leaf:
            return []
        nodes = [(path, node)]
        for key, child in node.children.items():
            nodes += get_internal_nodes(child, path + (key,))
        return nodes

    def get_node_by_path(node, path):
        """Retrieve a node given its path (sequence of keys)."""
        for key in path:
            node = node.children[key]
        return node

    def prune_node(node):
        """Convert internal node to leaf using majority of its subtree labels."""
        node.is_leaf = True
        node.split_attr = None
        node.split_value = None
        node.children = {}
        node.prediction = node.prediction  # Keep previous prediction (already majority at build time)

    def post_prune(tree, X_val, y_val):
        """Post-prune the tree using validation set accuracy."""
        best_acc = np.mean(predict(X_val, tree) == y_val)
        acc_list = [best_acc]
        node_counts = [count_nodes(tree)]

        while True:
            pruned = False
            candidates = get_internal_nodes(tree)

            for path, node in candidates:
                parent = get_node_by_path(tree, path[:-1]) if path else None
                backup = node.children.copy(), node.split_attr, node.split_value, node.is_leaf

                # Prune node
                node.is_leaf = True
                node.children = {}
                node.split_attr = None
                node.split_value = None
                node.prediction = majority_class(y_val)

                new_acc = np.mean(predict(X_val, tree) == y_val)
                if new_acc >= best_acc:
                    best_acc = new_acc
                    acc_list.append(best_acc)
                    node_counts.append(count_nodes(tree))
                    pruned = True
                    break  # restart pruning loop

                # Undo prune
                node.children, node.split_attr, node.split_value, node.is_leaf = backup

            if not pruned:
                break  # No more beneficial pruning

        return acc_list, node_counts

    def count_nodes(node):
        """Count total nodes in tree (used for plotting)."""
        if node.is_leaf:
            return 1
        return 1 + sum(count_nodes(child) for child in node.children.values())
    
    
    tree = build_tree(train_X_encoded, train_y, max_depth=55)
    acc_list, node_counts = post_prune(tree, valid_X_encoded, valid_y)
    test_preds = predict(test_X_encoded, tree)
    
    #save prediction as csv
    output_path = os.path.join(output_folder_path, "prediction_c.csv")
    pd.DataFrame({'prediction':test_preds}).to_csv(output_path,index=False) 
    
# ==== Part D: Varying Criteria ====
def part_d():
    """Compare entropy and gini decision trees."""
    # Combine all datasets temporarily for consistent encoding
    combined = pd.concat([train_X, valid_X, test_X], keys=['train', 'valid', 'test'])

    # One-hot encode all categorical columns (with >2 categories)
    combined_encoded = pd.get_dummies(combined)

    # Split back into individual sets
    train_X_encoded = combined_encoded.xs('train')
    valid_X_encoded = combined_encoded.xs('valid')
    test_X_encoded = combined_encoded.xs('test')
    
    clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.001, random_state=42)
    clf.fit(train_X_encoded, train_y)
    
    test_preds = clf.predict(test_X_encoded)
    
    #save prediction as csv
    output_path = os.path.join(output_folder_path, "prediction_d.csv")
    pd.DataFrame({'prediction':test_preds}).to_csv(output_path,index=False) 
    
# ==== Part E: Random Forest (Grid Search) ====
def part_e():
    """Train and tune a Random Forest."""
    combined = pd.concat([train_X, valid_X, test_X], keys=['train', 'valid', 'test'])

    # One-hot encode all categorical columns (with >2 categories)
    combined_encoded = pd.get_dummies(combined)

    # Split back into individual sets
    train_X_encoded = combined_encoded.xs('train')
    valid_X_encoded = combined_encoded.xs('valid')
    test_X_encoded = combined_encoded.xs('test')
    
    clf = RandomForestClassifier(
        criterion='entropy',
        oob_score=True,
        bootstrap=True,
        random_state=42,
        n_estimators=350,
        max_features=0.7,
        min_samples_split=10
    )
    clf.fit(train_X_encoded, train_y)
    
    test_preds = clf.predict(test_X_encoded)
    
    #save prediction as csv
    output_path = os.path.join(output_folder_path, "prediction_e.csv")
    pd.DataFrame({'prediction':test_preds}).to_csv(output_path,index=False) 

# ==== Part F: Feature Importance ====
def part_f():
    print("didnt implement this part")

if __name__ == "__main__":
    

    # Check number of arguments
    if len(sys.argv) != 6:
        print("Usage: python decision_tree.py <train_data_path> <validation_data_path> <test_data_path> <output_folder_path> <question_part>")
        sys.exit(1)

    train_path = sys.argv[1]
    valid_path = sys.argv[2]
    test_path = sys.argv[3]
    output_folder_path = sys.argv[4]
    part = sys.argv[5].lower()
    
    # Load data
    train_X, train_y, valid_X, valid_y, test_X = load_data(train_path, valid_path, test_path)

    # Question-specific execution
    if part == 'a':
        part_a()

    elif part == 'b':
        part_b()

    elif part == 'c':
        part_c()

    elif part == 'd':
        part_d()

    elif part == 'e':
        part_e()
    
    elif part == 'f':
        part_f()

    else:
        print(f"Invalid question part: {part}. Use one of: a, b, c, d, e, f.")
