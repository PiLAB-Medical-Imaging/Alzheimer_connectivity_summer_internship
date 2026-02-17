""" This script takes a graph and computes a series of graph level metrics that characterise the network. As of now, these are all single value metrics"""

import numpy as np
import networkx as nx


#### Graph level analysis ####
def graph_level_metrics(graph, threshold=None):
    if isinstance(graph, np.ndarray):
        if threshold is not None: 
            graph = np.where(graph>threshold, graph, 0)
        graph = nx.from_numpy_array(graph)

    n_isolates = len(list(nx.isolates(graph)))

    graph.remove_nodes_from(list(nx.isolates(graph)))

    avg_node_connectivity = nx.average_node_connectivity(graph)
    density = nx.density(graph)
    global_clustering = nx.average_clustering(graph, weight="weight")

    if nx.is_connected(graph):
        diam = nx.diameter(graph)
    else:
        largest_cc = max(nx.connected_components(graph), key=len)
        diam = nx.diameter(graph.subgraph(largest_cc))

    density = nx.density(graph)

    degree= nx.degree(graph, weight="weight")

    mean_degree = np.mean(degree)

    return {
        "m_connectivity": avg_node_connectivity,
        "density": density,
        "diameter": diam,
        "global_clustering": global_clustering,
        "isolates": n_isolates,
        "degree": mean_degree
    }

#### Node level analysis ####
def node_level_analysis():
    pass

#### Edge level analysis ####
def edge_level_analysis():
    pass

#### Driving Function ####
def analyse_graph(adjacency_matrix):
    graph = nx.from_numpy_array(adjacency_matrix)
    graph_level = graph_level_metrics(graph)
    #node_level = node_level_analysis(graph)
    #edge_level = edge_level_analysis(graph)
    return graph_level #, node_level, edge_level