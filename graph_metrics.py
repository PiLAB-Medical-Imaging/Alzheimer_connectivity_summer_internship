""" This script takes a graph and computes a series of graph level metrics that characterise the network. As of now, these are all single value metrics"""

import numpy as np
import networkx as nx


#### Graph level analysis ####
def graph_level_metrics(graph):
    print("Before Calculation")
    print(type(graph))
    print("Nodes:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())
    print("Density:", nx.density(graph))

    n_isolates = len(list(nx.isolates(graph)))

    graph.remove_nodes_from(list(nx.isolates(graph)))

    avg_node_connectivity = nx.average_node_connectivity(graph)
    density = nx.density(graph)
    global_clustering = nx.average_clustering(graph)

    if nx.is_connected(graph):
        diam = nx.diameter(graph)
    else:
        largest_cc = max(nx.connected_components(graph), key=len)
        diam = nx.diameter(graph.subgraph(largest_cc))

    density = nx.density(graph)

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    print("after Calculation")
    print("N:", n, "M:", m, "density:", density)
    
    
    return {
        "m_connectivity": avg_node_connectivity,
        "density": density,
        "diameter": diam,
        "global_clustering": global_clustering,
        "isolates": n_isolates
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