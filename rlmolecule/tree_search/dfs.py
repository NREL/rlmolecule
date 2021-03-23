import textwrap


def dfs(visited: set, node: 'GraphSearchState', parent: 'GraphSearchState') -> None:
    """ A depth-first search recursion of the node's children, yielding a GraphCycleError if the parent is found
    within the descendants of node.

    :param visited: initialized with an empty set, this builds recursively.
    :param node: The child node from which to search
    :param parent: The parent node which should not be included in child's reachable states.
    """
    if node not in visited:

        visited.add(node)

        if node == parent:
            raise GraphCycleError(parent)

        if node.children is not None:
            for child in node.children:
                dfs(visited, child, parent)


class GraphCycleError(Exception):
    def __init__(self, parent_node):
        self.parent_node = parent_node
        super().__init__()

    def __str__(self):
        return (f'Cyclic edge encountered at state:\n{self.parent_node}.\n' +
                textwrap.dedent("""
                This library is currently only equipped to search spaces defined as handle
                directional acyclic graphs. Cycles inside the search graph give rise to a
                graph history interaction problem. For more information, see:
                
                Childs, B. E., Brodeur, J. H., & Kocsis, L. (2008). Transpositions and move
                groups in Monte Carlo tree search. 2008 IEEE Symposium On Computational 
                Intelligence and Games. doi:10.1109/cig.2008.5035667
                
                One way to solve this error for search spaces with recurring positions is
                to add a "time" parameter which tracks the number of moves that have been
                made. (position, number_of_moves) pairs are therefore unlikely to recur as
                long as number_of_moves always increases.
                """))