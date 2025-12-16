""" This script takes a graph and computes a series of graph level metrics that characterise the network. As of now, these are all single value metrics"""

import numpy as np
import networkx as nx


#### Graph level analysis ####
def graph_level_metrics(graph):
    avg_node_connectivity = nx.average_node_connectivity(graph)
    density = nx.density(graph)
    diam = nx.diameter(graph)
    global_clustering = nx.average_clustering(graph)

    # Store as a dictionary and return

    metric_dict = {"m_connectivity": avg_node_connectivity, 
                   "density": density,
                   "diameter":diam,
                   "global_clustering": global_clustering}
    
    return metric_dict


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