"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    distance_v2 = distances.copy()

    nodes = [Node(name, None, 0.0, None, 0.0) for name in labels]

    node_count = 0

    # creating Q matrix

    while len(nodes) > 2:

        n = len(nodes)

        row_sums = np.sum(distance_v2, axis=1)
        
        Q_matrix = np.full((n, n), np.inf)

        val_1 = n - 2

        # determining distance for Q matrix
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                Q_matrix[i][j] = val_1 * distance_v2[i][j] - row_sums[i] - row_sums[j]

        # finding the minimum Q-value
        i, j = np.unravel_index(np.argmin(Q_matrix), Q_matrix.shape)

        dist_i = 0.5 * distance_v2[i][j] + (row_sums[i] - row_sums[j]) / (2 * val_1)
        dist_j = distance_v2[i][j] - dist_i

        parent_node = Node(f"node_{node_count}", nodes[i], dist_i, nodes[j], dist_j)

        node_count += 1

        remaining_node_indices = [k for k in range(n) if k != i and k != j]

        new_distances = []
        for k in remaining_node_indices:
                new_dist = (distance_v2[i][k] + distance_v2[j][k] - distance_v2[i][j]) / 2
                new_distances.append(new_dist)

        updated_matrix = distance_v2[np.ix_(remaining_node_indices, remaining_node_indices)]

        updated_size = n - 1

        new_matrix = np.zeros((updated_size, updated_size), dtype=float)

        new_matrix[-1, :-1] = new_distances
        new_matrix[:-1, -1] = new_distances
        new_matrix[:-1, :-1] = updated_matrix

        distance_v2 = new_matrix       

        nodes.pop(j)
        nodes.pop(i)
        nodes.append(parent_node)

    final_dist = distance_v2[0][1]

    root = Node("root", nodes[0], final_dist / 2, nodes[1], final_dist / 2)

    return root


def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """
    if ax is None:
        _, ax = plt.subplots()

    label_colors = kwargs.get("label_colors", {})

    fontsize = kwargs.get("fontsize", 8)
    # Counter used to give each leaf a different y-position
    leaf_counter = [0]

    def draw_node(node, x_pos):
        """
        Recursively draw the subtree rooted at node.

        Returns
        -------
        float
            The y-coordinate of this node.
        """

        # -----------------------------
        # Base case: leaf node
        # -----------------------------
        if node.left is None and node.right is None:
            y_pos = leaf_counter[0]
            leaf_counter[0] += 1
            colour = label_colors.get(node.name, "black")

            # Add leaf label
            ax.text(
                x_pos,
                y_pos,
                f" {node.name}",
                va="center"
                color=colour
                fontsize=fontsize,
            )

            return y_pos

        # -----------------------------
        # Recursive case: internal node
        # -----------------------------

        # Determine x-coordinate of children using branch lengths
        left_x = x_pos + node.left_distance
        right_x = x_pos + node.right_distance

        # Recursively obtain children's y-coordinates
        left_y = draw_node(node.left, left_x)
        right_y = draw_node(node.right, right_x)

        # Parent sits halfway between children vertically
        y_pos = (left_y + right_y) / 2

        left_colour = "black"
        right_colour = "black"

        if node.left.left is None and node.left.right is None:
            left_colour = label_colors.get(node.left.name, "black")

        if node.right.left is None and node.right.right is None:
            right_colour = label_colors.get(node.right.name, "black")

        # Horizontal branch to left child
        ax.plot(
            [x_pos, left_x],
            [left_y, left_y],
            color=left_colour
        )

        # Horizontal branch to right child
        ax.plot(
            [x_pos, right_x],
            [right_y, right_y],
            color=right_colour
        )

        # Vertical branch connecting the two children
        ax.plot(
            [x_pos, x_pos],
            [left_y, right_y],
            color="black"
        )

        return y_pos

    # Start drawing from the root at x = 0
    draw_node(tree, 0.0)

    # Axis formatting
    ax.set_xlabel("Genetic distance")
    ax.set_ylabel("")
    ax.set_yticks([])

    # Remove unnecessary borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return ax

def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """

    def count_and_sort(node):
        # Leaf
        if node is None:
            return 0

        if node.left is None and node.right is None:
            return 1

        left_count = count_and_sort(node.left)
        right_count = count_and_sort(node.right)

        # Put subtree with fewer leaves on the left
        if left_count > right_count:
            node.left, node.right = node.right, node.left

            node.left_distance, node.right_distance = (
                node.right_distance,
                node.left_distance,
            )

            left_count, right_count = right_count, left_count

        return left_count + right_count

    count_and_sort(tree)


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()