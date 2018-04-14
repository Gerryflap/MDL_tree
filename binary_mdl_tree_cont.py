import numpy as np
from scipy.special import comb

lg = np.log2


def L(n, k, b):
    return lg(b + 1) + lg(comb(n, k, exact=True))


class DTLeaf(object):
    def __init__(self, parent, input_data, labels, attr_continuous, max_depth, c):
        self.input_data = input_data
        self.labels = labels
        self.parent = parent
        self.max_depth = max_depth
        self.attr_continuous = attr_continuous
        self.c = c

        # Count the number of each class
        self.class_counts = np.zeros((2,))
        self.class_counts[1] = np.sum(labels)
        self.class_counts[0] = labels.shape[0] - self.class_counts[1]

        self.leaf_class = np.argmax(self.class_counts)
        self.exception_cost = self._calculate_exception_cost()

    def _calculate_exception_cost(self):
        """
        The exception cost can be calculated using:
            n = number of samples
            k = number of exceptions (label != leaf class)
            b = maximum number of possible exceptions. Which is (n+1)/2 for binary classification according to the paper.
                (n+1)/2 is interpreted as an integer division which is // in python3.

        """
        n = self.labels.shape[0]
        k = self.class_counts[1 - self.leaf_class]
        b = (n + 1) // 2
        exception_cost = L(n, k, b)
        return exception_cost

    def get_cost(self):
        """
        Returns the encoding cost of this leaf. This is calculated as the sum of:
            The preceding 0-bit indicating that this is a leaf node in the proposed encoding of the tree
            The default class of the leaf, which would only take one bit in a binary classification problem
            The length required to encode the exceptions, L(n, k, b) described elsewhere in this class.
            This is done times c according to the paper, to allow for larger trees.
        :return: The total encoding cost of this specific leaf
        """
        return 2 + self.c * self.exception_cost

    def get_recursive_cost(self):
        """
        Gives the cost of this node and descendants. Since a leaf has no descendants this will be simple this nodes cost.
        :return: The cost of this leaf node using get_cost()
        """
        return self.get_cost()

    def predict(self, x):
        return self.leaf_class

    def split(self):
        """
        Create a Node that can replace this leaf node.
        This will be the node that has the lowest MDL, even if it is higher than the MDL of this Leaf
        :return: A decision node that should replace the Leaf
        """
        if self.max_depth == 0:
            return None

        min_mdl = 0
        best_node = None
        for attr_index in range(self.input_data.shape[1]):
            node = DTNode(self.parent, attr_index, self.input_data, self.labels, self.attr_continuous, self.max_depth, self.c)
            cost = node.get_recursive_cost()
            if best_node is None or min_mdl > cost:
                best_node = node
                min_mdl = cost
        return best_node

    def expand(self):
        replacement = self.split()
        if replacement is not None:
            if self.parent is not None:
                self.parent.replace(self, replacement)
            replacement.expand()
            return replacement
        else:
            return self

    def prune(self, verbose=False):
        # The leafs cannot be pruned away
        return True

    def __str__(self):
        return "<Leaf c:%d N:%d>"%(self.leaf_class, self.input_data.shape[0])


class DTNode(object):
    def __init__(self, parent, attr_index, input_data, labels, attr_continuous, max_depth, c):
        """
        Initializes a decision node.
        :param attr_continuous: A-d' long boolean array specifying which attribute is continuous
        :param attr_index: The attribute to split on
        :param input_data: The input data, an Nx(A-d') numpy array containing the UNTESTED attributes for N samples
        :param labels: The labels for these N samples
        """
        self.input_data = input_data
        self.labels = labels
        self.parent = parent
        self.attr_index = attr_index
        self.split_data = {}
        self.max_depth = max_depth
        self.attr_continuous = attr_continuous
        self.c = c
        self.continuous = attr_continuous[attr_index]

        if self.continuous:
            self.threshold = self._find_threshold()

        N = input_data.shape[0]
        for i in range(N):
            x, label = input_data[i], labels[i]
            # TODO: Implement more continuous stuff
            attribute_value = x[attr_index]
            if attribute_value in self.split_data:
                # Add the input data row to the input data of the specific child without the already tested attribute x[attr_index]
                self.split_data[attribute_value][0].append(np.delete(x, attr_index))
                self.split_data[attribute_value][1].append(label)
            else:
                self.split_data[attribute_value] = (
                    [np.delete(x, attr_index)],
                    [label]
                )
        self.children = {}
        for k, v in self.split_data.items():
            self.children[k] = DTLeaf(self, np.array(v[0]), np.array(v[1]), attr_continuous, max_depth-1, c)

        # Count the number of each class
        self.class_counts = np.zeros((2,))
        self.class_counts[1] = np.sum(labels)
        self.class_counts[0] = labels.shape[0] - self.class_counts[1]

    def get_cost(self):
        """
        Return the cost of only this node.
        This is calculated a the sum of:
            The one bit encoding a decision node in the tree
            The number of bits necessary to encode the attribute: lg(A - d')
        :return: The cost of encoding this decision node
        """
        values = self.input_data[:, self.attr_index]
        m = len(set(values))
        return 1 + lg(self.input_data.shape[1]) + lg(m**0.5) if self.continuous else 0

    def get_recursive_cost(self):
        """
        Calculate the cost of this node and all descendants of this node
        :return: The sum of all costs from this node downwards
        """
        cost = self.get_cost()
        for _, child in self.children.items():
            cost += child.get_recursive_cost()
        return cost

    def predict(self, x):
        attr_value = x[self.attr_index]
        child = self.children[attr_value]
        child.predict(np.delete(x, self.attr_index))

    def get_children(self):
        return list(self.children.values())

    def expand(self):
        """
        Expands all child nodes
        """
        for child in self.children.values():
            child.expand()

    def replace(self, child, new_child):
        key = None
        for k, v in self.children.items():
            if v == child:
                key = k
                break
        self.children[key] = new_child

    def prune(self, verbose=False):
        """
        Check if this node has to be pruned.
        This happens in 2 steps:
            1. Recursively call this on all children
            2. If the children return true, meaning they're either a leaf or will became a leaf,
                check if this node needs pruning
        :return: true if this node is pruned (or a leaf), false if not
        """
        any_pruned = False
        for key, child in self.children.items():
            if type(child) == DTNode:
                pruned = child.prune(verbose)
                any_pruned = any_pruned or pruned
                if pruned:
                    self.children[key] = DTLeaf(self, child.input_data, child.labels, self.attr_continuous, self.max_depth-1, self.c)
            else:
                any_pruned = True
        if any_pruned:
            own_cost = self.get_recursive_cost()
            leaf = DTLeaf(self.parent, self.input_data, self.labels, self.attr_continuous, self.max_depth, self.c)
            pruned_cost = leaf.get_cost()
            if pruned_cost < own_cost:
                if verbose:
                    print("Reducing cost with %f bits by pruning %s"%(own_cost - pruned_cost, self))
                return True
        return False

    def __str__(self):
        child_str = ""
        for child in self.children.values():
            child_str += str(child) + " "
        return "<Node x[%d] %s>"%(self.attr_index, child_str)

    def _find_threshold(self):
        """
        Quoting the paper:
            "A second approach is to select approximately sqrt(m)
            (where m denotes the number of distinct values) evenly spaced values from the sorted list of values
            , and to use lg(sqrt(m)) bits to indicate which one to use as a cut-point.
             "
        :return: The used threshold value
        """
        values = self.input_data[:, self.attr_index]
        m = len(set(values))

        sort = np.argsort(values)
        s_values = values[sort]
        s_labels = self.labels[sort]
        threshold_values = np.mean(np.array([s_values[:-1], s_values[1:]]), axis=0)
        # Pick only sqrt(m) distinct values:
        step_size = s_values.shape[0]//(m**0.5)
        threshold_values = threshold_values[::step_size]

        n_class_1 = np.sum(self.labels)
        n_class_0 = self.labels.shape[0] - n_class_1

        min_cost = 0
        min_threshold = None

        for i, t in enumerate(threshold_values):
            label_index = i*step_size
            classes_left = np.zeros((2,))
            classes_left[1] = np.sum(s_labels[:label_index+1])
            classes_left[0] = label_index - classes_left[1]

            classes_right = np.zeros((2,))
            classes_right[1] = n_class_1 - classes_left[1]
            classes_right[0] = n_class_0 - classes_left[0]

            class_left = np.argmax(classes_left)
            class_right = np.argmax(classes_right)

            # Cost of left leaf:
            n = label_index  # Number of total values up to the threshold
            k = classes_left[1-class_left]  # Number of wrongly classified classes
            b = (n + 1) // 2
            exception_cost = L(n, k, b)

            # Cost of right leaf:
            n = s_labels.shape[0] - label_index  # Number of total values above or equal to the threshold
            k = classes_right[1-class_right]  # Number of wrongly classified classes
            b = (n + 1) // 2
            exception_cost += L(n, k, b)

            if min_threshold is None or exception_cost < min_cost:
                min_threshold = t
                min_cost = exception_cost
        return min_threshold





class BinaryContinuousMDLTreeClassifier(object):
    """
        This class models a Minimum Description Length (MDL) based Decision Tree Classifier For 2 Classes
    """

    def __init__(self, c, max_depth=100):
        """
        Creates a new instance of the MDL tree
        :param c: The c constant defines the ratio c_d / c_t .
            A higher value of c means that a larger tree is favored over more exceptions.
        """
        self.c = c
        self.tree = None
        self.max_depth = max_depth

    def fit(self, input_data, labels, attr_continuous, verbose=False, print_full_pruning=False):
        """
        Induces a decision tree on the given data.
        This method can only be called once.
        :param attr_continuous: Defines which attributes are continuous
        :param print_full_pruning: Print full pruning process
        :param verbose: Print extra information
        :param input_data: An NxA numpy array (Matrix) with N samples each having A attributes
        :param labels: An N long numpy array containing the output class of each sample in input_data
        """
        tree = DTLeaf(None, input_data, labels, attr_continuous, self.max_depth, self.c)
        tree = tree.expand()
        self.tree = tree
        self._prune(verbose, print_full_pruning)


    def predict(self, input_data):
        """
        Classifies the given data according to the fitted tree
        :param input_data: An NxA numpy array (Matrix) with N samples each having A attributes
        :return: An N long numpy array containing the predicted class for each of the given samples
        """
        prediction = []
        for x in input_data:
            prediction.append(self.tree.predict(x))
        return np.array(prediction)

    def _prune(self, verbose=False, print_full=False):
        if verbose:
            print("Tree cost before pruning: ", self.tree.get_recursive_cost())
        pruned = self.tree.prune(print_full)
        if pruned:
            self.tree = DTLeaf(None, self.tree.input_data, self.tree.labels, self.tree.attr_continuous, self.max_depth, self.c)
        if verbose:
            print("Tree cost after pruning: ", self.tree.get_recursive_cost())

    def __str__(self):
        return str(self.tree)
