from binary_mdl_tree_cont import BinaryContinuousMDLTreeClassifier
import numpy as np

data = np.genfromtxt("iris.csv", delimiter=',')
print(data)
X = data[:, :-1]
Y = data[:, -1]

X0 = X[Y==0]
X1 = X[Y==1]

# Split off the first half of both class 0 and 1 for training
train_X = np.concatenate((X0[:X0.shape[0]//2], X1[:X1.shape[0]//2]))
train_Y = np.concatenate((np.zeros((X0.shape[0]//2, )), np.zeros((X1.shape[0]//2, ))))

# And the second half for testing
test_X = np.concatenate((X0[X0.shape[0]//2:], X1[X1.shape[0]//2:]))
test_Y = np.concatenate((np.zeros((X0.shape[0]//2, )), np.zeros((X1.shape[0]//2, ))))


# Remove all class = 0 from the dataset, because we have a binary classifier
# Class 0 is too easy to separate
X = X[Y != 0]
Y = Y[Y != 0] - 1 # Change class 1 -> 0 and class 2 -> 1

tree = BinaryContinuousMDLTreeClassifier(1)
tree.fit(X, Y, np.array([True]*4), verbose=True)
print(tree.predict(X))
print("Accuracy: ", np.sum(tree.predict(X) == Y)/X.shape[0])
print(tree)

