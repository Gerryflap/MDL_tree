from binary_mdl_tree_cont import BinaryContinuousMDLTreeClassifier
import numpy as np
from sklearn import tree as sktree

data = np.genfromtxt("iris.csv", delimiter=',')
print(data)
X = data[:, :-1]
Y = data[:, -1]

# Remove all class = 0 from the dataset, because we have a binary classifier
# Class 0 is too easy to separate
X = X[Y != 0]
Y = Y[Y != 0] - 1 # Change class 1 -> 0 and class 2 -> 1

# Split the dataset into the two classes so it can be evenly split in test and train
X0 = X[Y==0]
X1 = X[Y==1]

# Split off the first half of both class 0 and 1 for training
train_X = np.concatenate((X0[:X0.shape[0]//2], X1[:X1.shape[0]//2]))
train_Y = np.concatenate((np.zeros((X0.shape[0]//2, )), np.ones((X1.shape[0]//2, ))))

# And the second half for testing
test_X = np.concatenate((X0[X0.shape[0]//2:], X1[X1.shape[0]//2:]))
test_Y = np.concatenate((np.zeros((X0.shape[0]//2, )), np.ones((X1.shape[0]//2, ))))

print("MDL Tree: ")
tree = BinaryContinuousMDLTreeClassifier(1)
tree.fit(train_X, train_Y, np.array([True] * 4), verbose=True)
print("Train accuracy: ", np.sum(tree.predict(train_X) == train_Y) / train_X.shape[0])
print("Test accuracy: ", np.sum(tree.predict(test_X) == test_Y) / test_X.shape[0])
print("Tree size: ", len(tree))
print(tree)
print()


print("CART Tree: ")
tree_cart = sktree.DecisionTreeClassifier()
tree_cart.fit(train_X, train_Y)
print("Train accuracy: ", np.sum(tree_cart.predict(train_X) == train_Y) / train_X.shape[0])
print("Test accuracy: ", np.sum(tree_cart.predict(test_X) == test_Y) / test_X.shape[0])
print("Tree size: ", tree_cart.tree_.node_count)
print(tree_cart)
print()


for c in [2, 5, 10, 2000]:
    print("Tree for c = ", c)
    tree = BinaryContinuousMDLTreeClassifier(c)
    tree.fit(train_X, train_Y, np.array([True]*4), verbose=True)
    print("Train accuracy: ", np.sum(tree.predict(train_X) == train_Y)/train_X.shape[0])
    print("Test accuracy: ", np.sum(tree.predict(test_X) == test_Y)/test_X.shape[0])
    print(tree)
    print()

