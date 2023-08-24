# This module provides simple implementations of a few graph-analytical techniques.
# For convenience, we assume that the weighted adjacency matrix A has already been
# computed for any given graph. Note that this implies that the graph vertices have
# been ordered. Also note that we assume 'loss' graphs, such that edge-weight A_ij
# is interpreted as the score lost from vertex v_i to vertex v_j.

import numpy as np


def adjusted_scores(scores):
    """
    Reinterprets the one-against-all scores as one-against-one scores.

    Input:
        - scores (ndarray): The N-dim vector of scores.
    Returns:
        - adj_scores (ndarray): The N-dim vector of adjusted scores.
    """
    N = len(scores)
    adj_scores = np.zeros(N, dtype=float)
    for i, score in enumerate(scores):
        adj_scores[i] = score / (score + (1 - score) / (N - 1))
    return adj_scores


def flow_prestige(A):
    """
    Computes the flow-prestige score for each of the N vertices
    in the graph. These scores are non-negative and sum to unity.
    See notebooks/B_graph_analytics.ipynb for more details.
    
    The graph edges specify loss-rate, such that the weight A_ij
    of edge v_i -> v_j indicates the average transfer per unit time
    of score from vertex v_i to vertex v_j. This transfer represents
    a loss of prestige by vertex v_i and a gain of prestige by
    vertex v_j.
    
    Inputs:
        - A (ndarray): The N x N weighted adjacency matrix.
    Returns:
        - scores (ndarray): The N-dim vector of prestige scores.
    """
    # Compute the flow-rate matrix R of the graph.
    R = A.T.copy()
    s = np.sum(R, axis=0)
    np.fill_diagonal(R, -s)
    # Find eigenvector with zero eigenvalue
    w, v = np.linalg.eig(R)
    idx = np.where(np.abs(w) <= 1e-6)[0][0]
    # Obtain normalised scores
    scores = np.abs(np.real(v[:, idx]))
    scores /= np.sum(scores)
    return scores
