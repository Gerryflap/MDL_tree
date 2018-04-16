from binary_mdl_tree_cont import BinaryContinuousMDLTreeClassifier, categorical_to_int
import numpy as np
from sklearn import tree as sktreelib

data = np.genfromtxt("agaricus-lepiota.csv", delimiter=',', dtype=None)
X = data[:, 1:]
Y = data[:, 0]

# Split the dataset into the two classes so it can be evenly split in test and train
classes = np.unique(Y)
print("Training for classes: ", classes)
X0 = X[Y==classes[0]]
X1 = X[Y==classes[1]]

np.random.shuffle(X0)
np.random.shuffle(X1)

N_train_samples = 50

# Split off the first half of both class 0 and 1 for training
train_X = np.concatenate((X0[:N_train_samples], X1[:N_train_samples]))
train_Y = np.concatenate((np.zeros((N_train_samples, )), np.ones((N_train_samples, ))))

# And the second half for testing
test_X = np.concatenate((X0[N_train_samples:], X1[N_train_samples:]))
test_Y = np.concatenate((np.zeros((X0.shape[0] - N_train_samples, )), np.ones((X1.shape[0] - N_train_samples, ))))

# Convert the strings to integer features for CART
tr_X = categorical_to_int(train_X)
te_X = categorical_to_int(test_X)


print("MDL tree:")
tree = BinaryContinuousMDLTreeClassifier(1)
tree.fit(train_X, train_Y, np.array([False]*X.shape[1]), verbose=True, print_full_pruning=False)
print("Train accuracy: ", np.sum(tree.predict(train_X) == train_Y)/train_X.shape[0])
print("Test accuracy: ", np.sum(tree.predict(test_X) == test_Y)/test_X.shape[0])
print("Tree size: ", len(tree))
print(tree)
print()


print("CART tree:")
tree_cart = sktreelib.DecisionTreeClassifier()
tree_cart.fit(tr_X, train_Y)
print("Train accuracy: ", np.sum(tree_cart.predict(tr_X) == train_Y)/train_X.shape[0])
print("Test accuracy: ", np.sum(tree_cart.predict(te_X) == test_Y)/test_X.shape[0])
print("Tree size: ", tree_cart.tree_.node_count)

# for c in [0.01, 0.5, 1, 3, 5, 100]:
#     print("Tree for c = ", c)
#     tree = BinaryContinuousMDLTreeClassifier(c)
#     tree.fit(train_X, train_Y, np.array([False]*X.shape[1]), verbose=True, print_full_pruning=False)
#     print("Train accuracy: ", np.sum(tree.predict(train_X) == train_Y)/train_X.shape[0])
#     print("Test accuracy: ", np.sum(tree.predict(test_X) == test_Y)/test_X.shape[0])
#     print("Tree size: ", len(tree))
#     print(tree)
#     print()

