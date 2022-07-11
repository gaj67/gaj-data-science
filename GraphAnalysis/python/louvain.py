import numpy as np


class GraphClustering:
    """
    Implements Louvain clustering (or community detection) for
    a directed graph.
    """

    def __init__(self, A, direction=0):
        """
        Initialises the clusters from the graph adjacency matrix.

        Inputs:
            - A (array): The non-negatively weighted graph adjacency matrix.
            - direction (int): A flag indicating whether to follow
                out-edges (if positive), or in-edges (if negative), or both
                out-edges and in-edges (if zero).
        """
        self._Aij = A
        self._Aid = A_i_dot = np.sum(A, axis=1)
        self._Adj = A_dot_j = np.sum(A,axis=0)
        self._Add = A_dot_dot = np.sum(A_i_dot)
        self._num_vertices = V = len(A)
        # vertex -> cluster
        self._clusters = [{vertex} for vertex in range(V)]
        self._score = (
            np.trace(A) - np.sum(A_i_dot * A_dot_j) / A_dot_dot
        ) / A_dot_dot
        self._neighbours = (
            self._out_neighbours if direction > 0
            else self._in_neighbours if direction < 0
            else self._all_neighbours
        )

    def clusters(self):
        """
        Obtains the currently assigned clusters (in arbitrary order).

        Returns:
            - clusters (set of tuple): The collection of clusters of
                vertex indices.
        """
        return {tuple(cluster) for cluster in self._clusters}

    def score(self):
        """
        Obtains the modularity score of the currently assigned clusters.

        Returns:
            - score (float): The modularity score.
        """
        return self._score

    def _out_neighbours(self, vertex):
        """
        Obtains the destination vertices for the given source vertex.

        Input:
            - vertex (int): The source vertex index.
        Returns:
            - neighbours (array): The neighbour indices.
        """
        return np.where(self._Aij[vertex,:] > 0)[0]

    def _in_neighbours(self, vertex):
        """
        Obtains the source vertices for the given destination vertex.

        Input:
            - vertex (int): The destination vertex index.
        Returns:
            - neighbours (array): The neighbour indices.
        """
        return np.where(self._Aij[:,vertex] > 0)[0]

    def _all_neighbours(self, vertex):
        """
        Obtains both source vertices directed into the given vertex,
        and destination vertices directed out from the given vertex.

        Input:
            - vertex (int): The vertex index.
        Returns:
            - neighbours (array): The neighbour indices.
        """
        return np.where(
            (self._Aij[:,vertex] > 0) | (self._Aij[vertex,:] > 0)
        )[0]

    def _sum(self, rows, columns):
        """
        Computes the sum of the adjacency sub-matrix specified by the given
        rows and columns.

        Inputs:
            - rows (list or tuple): The source vertex indices.
            - columns (list or tuple): The destination vertex indices.
        Returns:
            - sum (int or float): The sub-matrix sum.
        """
        return np.sum(self._Aij[np.ix_(rows, columns)])

    def _negate_cluster(self, cluster):
        """
        Obtains the vertices not in the given cluster.

        Input:
            - cluster (array): The indices of vertices in the cluster.
        Returns:
            - negation (array): The indices of vertices not in the cluster.
        """
        return set(range(self._num_vertices)) - cluster

    def _score_cluster(self, cluster):
        """
        Computes the modularity score of the given cluster.

        Input:
            - cluster (set): The indices of vertices in the cluster.
        Returns:
            - score (float): The modularity score.
        """
        if len(cluster) == 0:
            return 0.0
        if len(cluster) == 1:
            vertex = next(iter(cluster))
            return (
                self._Aij[vertex, vertex]
                - self._Aid[vertex] * self._Adj[vertex] / self._Add
            ) / self._Add
        t_cluster = tuple(cluster)
        internal_weight = self._sum(t_cluster, t_cluster)
        t_not_cluster = tuple(self._negate_cluster(cluster))
        out_weight = self._sum(t_cluster, t_not_cluster)
        in_weight = self._sum(t_not_cluster, t_cluster)
        S_out = internal_weight + out_weight
        S_in = internal_weight + in_weight
        return (internal_weight - S_out * S_in / self._Add) / self._Add

    def _score_move(self, vertex, from_cluster, to_cluster):
        """
        Computes the change in modularity score for potentially moving a
        vertex from one cluster to another. The move is not actually performed.
        The score will not change if the clusters are identical, i.e. the
        vertex doesn't change cluster. Assumes the vertex is in the cluster.

        Inputs:
            - vertex (int): The index of the vertex to move.
            - from_cluster (set): The indices of vertices in the cluster
                from which to move the vertex.
            - to_cluster (set): The indices of vertices in the cluster
                to which to move the vertex.
        Returns:
            - delta (float): The change in modularity score.
        """
        if to_cluster is from_cluster:
            return 0.0
        new_cluster1 = from_cluster - {vertex}
        new_cluster2 = to_cluster | {vertex}
        return (
            self._score_cluster(new_cluster1)
            - self._score_cluster(from_cluster)
            + self._score_cluster(new_cluster2)
            - self._score_cluster(to_cluster)
        )

    def _move(self, vertex, from_cluster, to_cluster):
        """
        Moves the given vertex from one cluster to another.
        The modularity score is not updated.
        Assumes the vertex is in the cluster.

        Inputs:
            - vertex (int): The index of the vertex to move.
            - from_cluster (set): The indices of vertices in the cluster
                from which to move the vertex.
            - to_cluster (set): The indices of vertices in the cluster
                to which to move the vertex.
        """
        from_cluster.remove(vertex)
        to_cluster.add(vertex)
        self._clusters[vertex] = to_cluster

    def _update(self):
        """
        Performs one pass through the graph, attempting to move each vertex
        from its cluster to a neighbouring cluster in order to maximise the
        increase in modularity score (if possible).

        Returns:
            - delta (float): The improvement in modularity score. This will be
                zero if no improvement could be found.
        """
        old_score = self._score
        for vertex, from_cluster in enumerate(self._clusters):
            max_delta = 0.0
            max_cluster = None
            for other in self._neighbours(vertex):
                to_cluster = self._clusters[other]
                delta = self._score_move(vertex, from_cluster, to_cluster)
                if delta > max_delta:
                    max_delta = delta
                    max_cluster = to_cluster
            if max_delta > 0.0:
                self._move(vertex, from_cluster, max_cluster)
                self._score += max_delta
        return self._score - old_score

    def optimise(self):
        """
        Repeatedly repartitions the vertices into clusters in order to
        approximately maximise the modularity score.

        Returns:
            - changed (bool): A flag indicating whether or not the clusters
                were repartitioned.
        """
        changed = False
        while self._update() > 0.0:
            changed = True
        return changed

    def aggregate(self):
        """
        Computes the coarse-grained adjacency matrix of the clustered graph.

        Returns:
            - A (array): The cluster-to-cluster adjacency matrix.
        """
        _clusters = self.clusters()
        C = len(_clusters)
        A = np.zeros((C, C), dtype=self._Aij.dtype)
        for i, from_cluster in enumerate(_clusters):
            for j, to_cluster in enumerate(_clusters):
                A[i,j] = self._sum(from_cluster, to_cluster)
        return A


def _denest(x):
    """
    Removes unnecessary tuple nesting.
    """
    while True:
        if isinstance(x, int):
            return x
        if len(x) > 1:
            return x
        x = x[0]


def taxonomy(A, direction=0):
    """
    Computes a hierarchical partitioning of the graph vertices.

    Inputs:
        - A (array): The non-negatively weighted graph adjacency matrix.
        - direction (int): A flag indicating whether to follow
            out-edges (if positive), or in-edges (if negative), or both
            out-edges and in-edges (if zero).
    Returns:
        - taxonomy (tuple): The vertex partition hierarchy.
    """
    levels = []
    while True:
        g = GraphClustering(A, direction)
        changed = g.optimise()
        levels.append(list(g.clusters()))
        if not changed:
            break
        A = g.aggregate()
    # Construct nested taxonomy
    taxonomy = levels[0]
    for level in levels:
        if level is taxonomy:
            continue
        taxonomy = _denest(tuple(
            _denest(tuple(_denest(taxonomy[idx]) for idx in cluster))
            for cluster in level
        ))
    return taxonomy


def _flatten(x):
    """
    Flattens nested tuples.
    """
    if isinstance(x, int):
        return x
    y = []
    for z in x:
        if isinstance(z, int):
            y.append(z)
        else:
            y.extend(_flatten(z))
    return tuple(y)


########## Test case #############
if __name__ == "__main__":
    # [1, 2, 3] <-w5- 0 -w1-> 4 <-w10- [5, 6]
    A = np.array([
        [0, 5, 5, 5, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 10, 0, 0],
        [0, 0, 0, 0, 10, 0, 0],
    ])
    print("Test: follow out-edges")
    t = taxonomy(A, direction=+1)
    print(t)
    # Expect 2 high-level clusters
    assert len(t) == 2
    # Expect split between vertices 0 and 1
    expected_cluster0 = {0, 1, 2, 3}
    expected_cluster1 = {4, 5, 6}
    for cluster in t:
        cluster = set(_flatten(cluster))
        if 0 in cluster:
            assert cluster == expected_cluster0
        else:
            assert cluster == expected_cluster1
    ################################
    print("Test: follow in-edges")
    t = taxonomy(A, direction=-1)
    print(t)
    # Expect 2 high-level clusters
    assert len(t) == 2
    # Expect split between vertices 0 and 1
    expected_cluster0 = {0, 1, 2, 3}
    expected_cluster1 = {4, 5, 6}
    for cluster in t:
        cluster = set(_flatten(cluster))
        if 0 in cluster:
            assert cluster == expected_cluster0
        else:
            assert cluster == expected_cluster1
    ################################
    print("Test: follow all edges (undirected)")
    t = taxonomy(A, direction=0)
    print(t)
    # Expect 2 high-level clusters
    assert len(t) == 2
    # Expect split between vertices 0 and 1
    expected_cluster0 = {0, 1, 2, 3}
    expected_cluster1 = {4, 5, 6}
    for cluster in t:
        cluster = set(_flatten(cluster))
        if 0 in cluster:
            assert cluster == expected_cluster0
        else:
            assert cluster == expected_cluster1
