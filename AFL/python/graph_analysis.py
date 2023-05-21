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
        - scores (vector): The N-dim array of scores.
    Returns:
        - adj_scores (vector): The N-dim array of adjusted scores.
    """
    N = len(scores)
    adj_scores = np.zeros(N, dtype=float)
    for i, score in enumerate(scores):
        adj_scores[i] = score / (score + (1 - score) / (N - 1))
    return adj_scores


def flow_prestige(A, M):
    """
    Computes the flow-prestige score for each of the N vertices
    in the graph. These scores are non-negative and sum to unity.
    
    Inputs:
        - A (matrix): The N x N array of total edge weights.
        - M (matrix): The N x N array of edge counts.
    Returns:
        - scores (vector): The N-dim array of scores.
    """
    # Compute the transpose of the flow-rate matrix R of the graph.
    R_T = np.zeros(A.shape, dtype=float)
    ind = M > 0
    R_T[ind] = A[ind] / M[ind]
    s = np.sum(R_T, axis=1)
    np.fill_diagonal(R_T, -s)
    # Find eigenvector with zero eigenvalue
    w, v = np.linalg.eig(R_T.T)
    idx = np.where(np.abs(w) <= 1e-6)[0][0]
    # Obtain normalised scores
    scores = np.real(v[:, idx])
    scores /= np.sum(scores)
    return scores
