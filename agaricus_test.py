from binary_mdl_tree_cont import BinaryContinuousMDLTreeClassifier
import numpy as np

data = np.genfromtxt("agaricus-lepiota.csv", delimiter=',', dtype=None)
print(data)
X = data[:, 1:]
Y = data[:, 0]

print(Y.shape)

# Split the dataset into the two classes so it can be evenly split in test and train
classes = np.unique(Y)
print("Training for classes: ", classes)
X0 = X[Y==classes[0]]
X1 = X[Y==classes[1]]

np.random.shuffle(X0)
np.random.shuffle(X1)

# Split off the first half of both class 0 and 1 for training
train_X = np.concatenate((X0[:X0.shape[0]//2], X1[:X1.shape[0]//2]))
train_Y = np.concatenate((np.zeros((X0.shape[0]//2, )), np.ones((X1.shape[0]//2, ))))

# And the second half for testing
test_X = np.concatenate((X0[X0.shape[0]//2:], X1[X1.shape[0]//2:]))
test_Y = np.concatenate((np.zeros((X0.shape[0]//2, )), np.ones((X1.shape[0]//2, ))))

print(train_X.shape, test_X.shape)

for c in [0.01, 0.5, 1, 3, 5, 100]:
    print("Tree for c = ", c)
    tree = BinaryContinuousMDLTreeClassifier(c)
    tree.fit(train_X, train_Y, np.array([False]*X.shape[1]), verbose=True, print_full_pruning=False)
    print("Train accuracy: ", np.sum(tree.predict(train_X) == train_Y)/train_X.shape[0])
    print("Test accuracy: ", np.sum(tree.predict(test_X) == test_Y)/test_X.shape[0])
    print("Tree size: ", len(tree))
    print(tree)
    print()

