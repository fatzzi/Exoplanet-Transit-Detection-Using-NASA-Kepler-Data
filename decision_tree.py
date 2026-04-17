import numpy as np

def entropy(y):
    # y is an array of labels e.g. [0, 1, 0, 0, 1]
    classes = np.unique(y)
    total = len(y)
    ent = 0.0
    for c in classes:
        p = np.sum(y == c) / total   # proportion of class c
        ent -= p * np.log2(p)        # entropy formula
    return ent

def information_gain(X_column, y, threshold):
    # entropy before the split
    parent_entropy = entropy(y)
    
    # split into two groups based on threshold
    left_mask = X_column <= threshold
    right_mask = X_column > threshold
    
    left_y = y[left_mask]
    right_y = y[right_mask]
    
    # if one side is empty, no gain
    if len(left_y) == 0 or len(right_y) == 0:
        return 0.0
    
    # weighted entropy after the split
    n = len(y)
    weighted_entropy = (len(left_y)/n) * entropy(left_y) + \
                       (len(right_y)/n) * entropy(right_y)
    
    # gain = how much entropy dropped
    return parent_entropy - weighted_entropy

def best_split(X, y):
    best_gain = 0.0
    best_feature = None
    best_threshold = None
    
    n_features = X.shape[1]
    
    for feature_index in range(n_features):
        X_column = X[:, feature_index]
        thresholds = np.unique(X_column)
        
        for threshold in thresholds:
            gain = information_gain(X_column, y, threshold)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold
    
    return best_feature, best_threshold

class Node:
    def __init__(self, feature=None, threshold=None, 
                 left=None, right=None, value=None):
        
        self.feature = feature       # which feature to split on
        self.threshold = threshold   # what value to split at
        self.left = left             # left subtree (≤ threshold)
        self.right = right           # right subtree (> threshold)
        self.value = value           # only set if this is a leaf node
    
    def is_leaf(self):
        return self.value is not None
    
def build_tree(X, y, depth=0, max_depth=10):
    n_samples = len(y)
    
    # STOPPING CONDITIONS (when to make a leaf)
    
    # 1. if all labels are the same, no point splitting
    if len(np.unique(y)) == 1:
        return Node(value=y[0])
    
    # 2. if we've hit max depth, return majority class
    if depth >= max_depth:
        majority_class = np.bincount(y).argmax()
        return Node(value=majority_class)
    
    # 3. if no samples left
    if n_samples == 0:
        return Node(value=0)
    
    # FIND BEST SPLIT
    best_feature, best_threshold = best_split(X, y)
    
    # if no good split found, return majority class
    if best_feature is None:
        majority_class = np.bincount(y).argmax()
        return Node(value=majority_class)
    
    # SPLIT THE DATA
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = X[:, best_feature] > best_threshold
    
    # RECURSIVELY BUILD LEFT AND RIGHT SUBTREES
    left_subtree = build_tree(X[left_mask], y[left_mask], 
                               depth+1, max_depth)
    right_subtree = build_tree(X[right_mask], y[right_mask], 
                                depth+1, max_depth)
    
    return Node(feature=best_feature, 
                threshold=best_threshold,
                left=left_subtree, 
                right=right_subtree)  
    
def predict_one(node, x):
    # if we've reached a leaf, return the answer
    if node.is_leaf():
        return node.value
    
    # otherwise ask the question at this node
    if x[node.feature] <= node.threshold:
        return predict_one(node.left, x)   # go left
    else:
        return predict_one(node.right, x)  # go right
    
def predict(node, X):
    return np.array([predict_one(node, x) for x in X])

# load data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val   = np.load('X_val.npy')
y_val   = np.load('y_val.npy')
X_test  = np.load('X_test.npy')
y_test  = np.load('y_test.npy')
X_candidates = np.load('X_candidates.npy')

# train
print("Training decision tree...")
tree = build_tree(X_train, y_train, max_depth=10)

# evaluate on validation
val_preds = predict(tree, X_val)
val_accuracy = np.mean(val_preds == y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# evaluate on test
test_preds = predict(tree, X_test)
test_accuracy = np.mean(test_preds == y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# predict on candidates
candidate_preds = predict(tree, X_candidates)
print(f"Candidate Predictions: {np.unique(candidate_preds, return_counts=True)}")

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("\n--- Test Set Evaluation ---")
print(classification_report(y_test, test_preds, 
      target_names=['CONFIRMED', 'FALSE POSITIVE']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_preds))
print(f"AUC-ROC: {roc_auc_score(y_test, test_preds):.4f}")
