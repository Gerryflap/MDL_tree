"""
    This is a test with the dataset used in the paper.

    Outlook: 0=sunny, 1=overcast, 2=rain
    Temperature: 0=hot, 1=mild, 2=cool
    Humidity: 0=high, 1=normal
    Windy: 0=false, 1=true

    Class: N=0, P=1
"""
from mdl_tree import BinaryMDLTreeClassifier
import numpy as np

data = np.array(
    [
        [0,0,0,0,0],    #1
        [0,0,0,1,0],    #2
        [1,0,0,0,1],    #3
        [2,1,0,0,1],    #4
        [2,2,1,0,1],    #5
        [2,2,1,1,0],    #6
        [1,2,1,1,1],    #7
        [0,1,0,0,0],    #8
        [0,2,1,0,1],    #9
        [2,1,1,0,1],    #10
        [0,1,1,1,1],    #11
        [1,1,0,1,1],    #12
        [1,0,1,0,1],    #13
        [0,1,0,1,0],    #14
    ]
)

X = data[:, :-1]
Y = data[:, -1]

for c in range(1, 10, 2):
    print("Tree for c=", c)
    tree = BinaryMDLTreeClassifier(c, max_depth=1000)
    tree.fit(X, Y, verbose=True, print_full_pruning=False)
    print(tree)
    print()

