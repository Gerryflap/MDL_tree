from binary_mdl_tree_cont import BinaryContinuousMDLTreeClassifier, categorical_to_int
import numpy as np
from sklearn import tree as sktreelib
import matplotlib.pyplot as plt

data = np.genfromtxt("agaricus-lepiota.csv", delimiter=',', dtype=None)
X = data[:, 1:]
Y = data[:, 0]

print(X.shape)

# Split the dataset into the two classes so it can be evenly split in test and train
classes = np.unique(Y)
print("Training for classes: ", classes)
X0 = X[Y==classes[0]]
X1 = X[Y==classes[1]]

cart_sum, mdl_sum = None, None

print(X0.shape)
for i in range(10):
    np.random.shuffle(X0)
    np.random.shuffle(X1)

    N_values = np.array([10, 12, 14, 16, 18, 20, 30, 50, 100, 200, 300, 400, 500])
    cart_values = []
    mdl_values = []

    for N_train_samples in N_values:
        print("N: ", N_train_samples)
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
        tr_acc = np.sum(tree.predict(train_X) == train_Y)/train_X.shape[0]
        te_acc = np.sum(tree.predict(test_X) == test_Y)/test_X.shape[0]
        l = len(tree)
        mdl_values.append([tr_acc, te_acc, l])


        print("CART tree:")
        tree_cart = sktreelib.DecisionTreeClassifier()
        tree_cart.fit(tr_X, train_Y)
        tr_acc = np.sum(tree_cart.predict(tr_X) == train_Y)/train_X.shape[0]
        te_acc = np.sum(tree_cart.predict(te_X) == test_Y)/test_X.shape[0]
        l = tree_cart.tree_.node_count
        cart_values.append([tr_acc, te_acc, l])


    mdl_values = np.array(mdl_values)
    cart_values = np.array(cart_values)
    if mdl_sum is None:
        mdl_sum = mdl_values
        cart_sum = cart_values
    else:
        mdl_sum += mdl_values
        cart_sum += cart_values

mdl_values = mdl_sum/10.0
cart_values = cart_sum/10.0
print(N_values)
print("MDL: ", mdl_values)
print("CART: ", cart_values)

plt.plot(N_values, mdl_values[:,1], label='MDL Tree')
plt.plot(N_values, cart_values[:,1], color='red', label='CART (sklearn)')
plt.xlabel("Training examples per class")
plt.ylabel("Test accuracy on remaining data")
plt.legend()
plt.show()

plt.plot(N_values, mdl_values[:,2], label='MDL Tree')
plt.plot(N_values, cart_values[:,2], color='red', label='CART (sklearn)')
plt.xlabel("Training examples per class")
plt.ylabel("Tree size (nodes)")
plt.legend()
plt.show()

# for c in [0.01, 0.5, 1, 3, 5, 100]:
#     print("Tree for c = ", c)
#     tree = BinaryContinuousMDLTreeClassifier(c)
#     tree.fit(train_X, train_Y, np.array([False]*X.shape[1]), verbose=True, print_full_pruning=False)
#     print("Train accuracy: ", np.sum(tree.predict(train_X) == train_Y)/train_X.shape[0])
#     print("Test accuracy: ", np.sum(tree.predict(test_X) == test_Y)/test_X.shape[0])
#     print("Tree size: ", len(tree))
#     print(tree)
#     print()

