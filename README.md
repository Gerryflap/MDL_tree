# MDL tree

An implementation of the MDL tree proposed in Quinlan, J. Ross, and Ronald L. Rivest. ”Inferring decision trees using the minimum
description length principle.” Information and computation 80.3 (1989)

The test files can be run using python3. The code depends on numpy and sklearn, please install them using pip, conda or manually. For the agaricus\_multi\_test.py matplotlib is also required for the graphs.

A description of the contents:
* agaricus\_multi\_test.py creates the average graphs for accuracy and treesize vs N samples
* agaricus\_small\_test.py runs both trees on a small part of the mushroom dataset
* agaricus_test.py runs both trees on the 50/50 split mushroom dataset
* iris_test.py runs both trees on the 50/50 split binary Iris dataset
* small_test.py runs the MDL tree algorithm on the small dataset
* binary\_mdl\_tree\_cont.py contains the BinaryContinuousMDLTreeClassifier which models the MDL tree algorithm from the paper. It can be imported and used in other python scripts.

The iris and agaricus-lepiota dataset are taken from the UCI machine learning repository.
They are only included to ease the use of this project. To acquire these or other datasets I advise you to use their website.
