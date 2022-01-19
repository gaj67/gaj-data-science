# Module for graph based algorithms

# Graph representation:
#  - An edges frame is a data frame with columns 'source.id', 'target.id' and 'weight',
#    where 'source' and 'target' each reference a vertex v_i in V (indexed from 1 to |V|),
#    and 'weight' gives the edge weight (often just 1 to measure degree).
#  - An edges representation is an object with a get_edges() method.
#  - A graph representation is an edges representation with the following methods:
#     * get_node_ids()           - returns a vector of all vertex identifiers.
#     * get_out_edges(source.id) - returns an edges frame of all edges directed from the source vertex.
#     * get_in_edges(target.id)  - returns an edges frame of all edges directed to the target vertex.

# Description:
#   Wraps an edges frame into an edges representation.
# Input:
#   edges - The edges frame.
# Output:
#   edges.repr - The edges representation.
graph.get_edges_repr = function(edges) {
    edges.repr = list()
    edges.repr$get_edges = function() {
        return(edges)
    }
    return(edges.repr)
}

# Description:
#   Wraps an edges edges representation into a graph representation.
# Input:
#   edges.repr - The edges representation.
# Output:
#   graph.repr - The graph representation.
graph.from_edges_repr = function(edges.repr) {
    edges = edges.repr$get_edges()
    graph.repr = list()
    graph.repr$get_edges = function() {
        return(edges)
    }
    graph.repr$get_node_ids = function() {
        return(sort(unique(c(edges$source.id, edges$target.id))))
    }
    graph.repr$get_out_edges = function(source.id) {
        return(edges[edges$source.id == source.id,])
    }
    graph.repr$get_in_edges = function(target.id) {
        return(edges[edges$target.id == target.id,])
    }
    return(graph.repr)
}

# Description:
#   Internal function to encapsulate all clustering information into a single object.
# Input:
#   num.nodes - The total number |V| of vertices in the graph, on the assumption that
#               every vertex id has been uniquely mapped to an integer from 1 to num.nodes.
#   trace     - An optional flag indicating whether (TRUE) or not (FALSE) to
#               track incremental score changes; defaults to FALSE.
# Output:
#   clusters - An object representing the internal cluster functions.
graph.init_clusters = function(num.nodes, trace=FALSE) {
    # Information about all nodes:
    # Sum of all edge weights (half of total vertex weight)
    total_edge_weights = 0
    # Map node index to cluster id
    node_cluster_map = 1:num.nodes
    # Count weight of internal edges in each cluster (both nodes in cluster)
    internal_cluster_edges_map = rep(0, num.nodes)
    # Count weight of external edges pointing into our out of each cluster (only one node in cluster)
    external_cluster_edges_map = rep(0, num.nodes)
    # Initial modularity score
    initial_score = 0
    # Cummulative change in modularity score
    change_score = 0
    if (trace) {
        change_scores = NULL
    }
    # Control initial score computation
    score_counter = 0
    
    # Information about about current (pivot) node:
    # Total edge weight of a node
    node_weight = 0
    # Weight of self edge to and from node
    self_weight = 0
    # Weights of edges from node to each cluster
    local_cluster_edges_map = rep(0, num.nodes)

    # Create object
    obj = list()
    # Add node v_i to new cluster if not already clustered
    obj$add_cluster_node = function(i) {
        # No action needed as nodes already indexed.
    }
    # Put node v_i into existing cluster c_i
    obj$set_cluster_node = function(i, c_i) {
        node_cluster_map[i] <<- c_i
    }
    # Add directed edge between node clusters
    obj$add_cluster_edge = function(i, j, w_ij) {
        if (i == j) w_ij = 0.5 * w_ij
        total_edge_weights <<- total_edge_weights + w_ij
        c_i = node_cluster_map[[i]]
        c_j = node_cluster_map[[j]]
        if (c_i == c_j) {
            internal_cluster_edges_map[c_i] <<- internal_cluster_edges_map[c_i] + w_ij
        } else {
            external_cluster_edges_map[c_i] <<- external_cluster_edges_map[c_i] + w_ij
            external_cluster_edges_map[c_j] <<- external_cluster_edges_map[c_j] + w_ij
        }
    }
    # Initialise local cluster edge counts for a single node
    obj$init_local_cluster_edges = function() {
        node_weight <<- 0
        self_weight <<- 0
        local_cluster_edges_map <<- rep(0, num.nodes)
    }
    # Add undirected edge between pivot node i and node j
    obj$add_local_cluster_edge = function(i, j, w_ij) {
        if (i == j) {
            self_weight <<- w_ij
            w_ij = 0.5 * w_ij
        }
        node_weight <<- node_weight + w_ij
        c_j = node_cluster_map[[j]]
        local_cluster_edges_map[c_j] <<- local_cluster_edges_map[c_j] + w_ij
    }
    # If possible, move pivot node to another cluster with highest positive change of score
    obj$move_to_best_cluster = function(i) {
        c_i = node_cluster_map[[i]]
        A_dd = 2 * total_edge_weights # A..
        A_id = node_weight # A_i.
        score_counter <<- score_counter + 1
        if (score_counter <= num.nodes) {
            # Continue computing initial score
            Q_i = (self_weight - A_id * A_id / A_dd) / A_dd
            initial_score <<- initial_score + Q_i
        }
        # Compute score change in removing node i from its cluster
        A_i_gpi = internal_cluster_edges_map[[c_i]] # A_i,g+i
        A_ig =  A_i_gpi - self_weight # A_i,g
        sigma_gpi_tot = 2 * internal_cluster_edges_map[[c_i]] + external_cluster_edges_map[[c_i]] # Sigma_g+i^tot
        sigma_tot = sigma_gpi_tot - A_id # Sigma_g^tot
        delta_Q_gpimi = -2 * (A_ig  - sigma_tot * A_id / A_dd) / A_dd # Delta Q_(g+i)-i
        max_delta_Q = -Inf
        max_cluster_id = 0
        local_cluster_edges_map[c_i] = 0 # Ignore current cluster
        for (c_j in which(local_cluster_edges_map > 0)) {
            sigma_tot = 2 * internal_cluster_edges_map[[c_j]] + external_cluster_edges_map[[c_j]] # Sigma_g'^tot
            A_igd = local_cluster_edges_map[[c_j]] # A_i,g'
            delta_Q_gdpi = 2 * (A_igd - sigma_tot * A_id / A_dd) / A_dd # Delta Q_g'+i
            delta_Q = delta_Q_gpimi + delta_Q_gdpi
            if (delta_Q > 1e-8 && delta_Q > max_delta_Q) {
                max_delta_Q = delta_Q
                max_cluster_id = c_j
                #print(paste0("DEBUG: Best delta-score=", delta_Q, ". Suggest move node from cluster ", c_i, " to cluster ", c_j))
            }
        }
        if (max_cluster_id > 0) {
            c_j = max_cluster_id
            change_score <<- change_score + max_delta_Q
            if (trace) {
                change_scores <<- c(change_scores, max_delta_Q)
            }
            # Remove node i from cluster c_i
            internal_cluster_edges_map[[c_i]] <<- internal_cluster_edges_map[[c_i]] - A_ig - 0.5 * self_weight
            external_cluster_edges_map[[c_i]] <<- external_cluster_edges_map[[c_i]] - A_id + 2 * A_ig + self_weight
            # Add node i to cluster c_j
            A_igd = local_cluster_edges_map[[c_j]] # A_i,g'
            internal_cluster_edges_map[[c_j]] <<- internal_cluster_edges_map[[c_j]] + A_igd + 0.5 * self_weight
            external_cluster_edges_map[[c_j]] <<- external_cluster_edges_map[[c_j]] + A_id - 2 * A_igd - self_weight
            node_cluster_map[i] <<- c_j
            return(TRUE)
        } else {
            return(FALSE)
        }
    }
    # Provide initial score for singleton clusters
    obj$get_inital_score = function() {
        return(initial_score)
    }
    # Provide final score for clustered graph
    obj$get_final_score = function() {
        return(initial_score + change_score)
    }
    # Provide incremental score changes
    obj$get_score_changes = function() {
        if (trace) return(change_scores)
        return(c())
    }
    # Remap the remaining clusters to indices in 1, 2, ..., K for smallest K
    obj$get_cluster_map = function() {
        cluster.nums = sort(unique(node_cluster_map))
        K = length(cluster.nums)
        new_cluster_map = rep(0, K)
        for (k in 1:K) {
            idx = which(node_cluster_map == cluster.nums[[k]])
            new_cluster_map[idx] = k
        }
        return(new_cluster_map)
    }
    # Expose object
    return(obj)
}

# Description:
#   An implementation of phase 1 of the Louvain modularity community detection (i.e. node clustering).
# Input:
#   graph.repr - A graph representation.
#   trace      - An optional flag indicating whether (TRUE) or not (FALSE) to
#                track incremental score changes; defaults to FALSE.
# Output:
#   clusters - A |V| length vector giving the cluster index c_i for each vertex v_i.
graph.cluster = function(graph.repr, trace=FALSE) {
    node.ids = graph.repr$get_node_ids()
    num.nodes = length(node.ids)
    # Initialisation:
    clusters = graph.init_clusters(num.nodes, trace)
    for (i in 1:num.nodes) {
        v_i = node.ids[i]
        clusters$add_cluster_node(i)
        # To avoid double counting, only traverse edges in source -> target direction.
        out_edges = graph.repr$get_out_edges(v_i)
        num.out_edges = nrow(out_edges)
        if (num.out_edges > 0) {
            for (edge.idx in 1:num.out_edges) {
                edge = out_edges[edge.idx,]
                v_j = edge$target.id
                j = which(node.ids == v_j)
                clusters$add_cluster_node(j)
                w_ij = edge$weight
                clusters$add_cluster_edge(i, j, w_ij)
            }
        }
    }
    
    # Cluster reassignments:
    while (TRUE) {
        changed = FALSE
        for (i in 1:num.nodes) {
            v_i = node.ids[i]
            clusters$init_local_cluster_edges()
            out_edges = graph.repr$get_out_edges(v_i)
            num.out_edges = nrow(out_edges)
            if (num.out_edges > 0) {
                for (edge.idx in 1:num.out_edges) {
                    edge = out_edges[edge.idx,]
                    v_j = edge$target.id
                    j = which(node.ids == v_j)
                    w_ij = edge$weight
                    clusters$add_local_cluster_edge(i, j, w_ij) # i -> j becomes i -- j.
                }
            }
            in_edges = graph.repr$get_in_edges(v_i)
            num.in_edges = nrow(in_edges)
            if (num.in_edges > 0) {
                for (edge.idx in 1:num.in_edges) {
                    edge = in_edges[edge.idx,]
                    v_j = edge$source.id
                    j = which(node.ids == v_j)
                    w_ji = edge$weight
                    clusters$add_local_cluster_edge(i, j, w_ji) # Yes, i <- j becomes i -- j.
                }
            }
            moved = clusters$move_to_best_cluster(i)
            changed = changed || moved
        }
        if (!changed) break
    }
    # Map node id to cluster id
    cluster.ids = clusters$get_cluster_map()
    names(cluster.ids) = node.ids
    if (trace) {
        scores = cumsum(c(clusters$get_inital_score(), clusters$get_score_changes()))
    } else {
        scores = c(clusters$get_inital_score(), clusters$get_final_score())
    }
    return(list(
        cluster.ids = cluster.ids,
        initial.score = clusters$get_inital_score(),
        final.score = clusters$get_final_score(),
        incremental.scores = scores
    ))
}

graph.matrix = function(graph.repr) {
    node.ids = graph.repr$get_node_ids()
    num.nodes = length(node.ids)
    D = matrix(data=0, nrow=num.nodes, ncol=num.nodes)
    for (i in 1:num.nodes) {
        v_i = node.ids[[i]]
        out.edges = graph.repr$get_out_edges(v_i)
        num.out.edges = nrow(out.edges)
        if (num.out.edges > 0) {
            for (edge.idx in 1:num.out.edges) {
                edge = out.edges[edge.idx,]
                v_j = edge$target.id
                j = which(node.ids == v_j)
                w_ij = edge$weight
                D[i, j] = D[i, j] + w_ij
            }
        }
    }
    rownames(D) = node.ids
    colnames(D) = node.ids
    return(D)
}

test.graph.cluster = function() {
    edges = data.frame(source.id=c("X", "Y", "Z"), target.id=c("Z", "X", "Y"), weight=1, stringsAsFactors=FALSE)
    edges.repr = graph.get_edges_repr(edges)
    graph.repr = graph.from_edges_repr(edges.repr)
    D = graph.matrix(graph.repr)
    A = D + t(D)
    diag(A) = diag(D)
    A_id = rowSums(A)
    A_dd = sum(A_id)
    Q_mat = (A - A_id %*% t(A_id) / A_dd) / A_dd
    # Step 0 clusters: 1, 2, 3
    delta0 = matrix(0, nrow=3, ncol=3)
    diag(delta0) = 1
    step0.score = sum(Q_mat * delta0)
    # Step 1 clusters: 2, 2, 3
    delta1 = delta0
    delta1[1, 2] = 1; delta1[2, 1] = 1
    step1.score = sum(Q_mat * delta1)
    # Step 2 clusters: 2, 2, 2
    delta2 = delta1
    delta2[1, 3] = 1; delta2[2, 3] = 1; delta2[3, 1] = 1; delta2[3, 2] = 1
    step2.score = sum(Q_mat * delta2)
    scores = c(step0.score, step1.score, step2.score)
    # Step 3 clusters: 1, 1, 1 (remapped)
    res = graph.cluster(graph.repr, trace=TRUE)
    print(paste0("Expected incremental scores: ", paste(scores, collapse=", ")))
    print(paste0("Observed incremental scores: ", paste(res$incremental.scores, collapse=", ")))
    print(paste0("Expected clusters: ", paste(rep(1, 3), collapse=", ")))
    print(paste0("Observed clusters: ", paste(res$cluster.ids, collapse=", ")))
}
