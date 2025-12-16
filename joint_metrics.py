import bids
import networkx as nx
import numpy as np

n = 100

connectome = np.random.rand(n,n)
fconn = np.random.rand(n,n)

def engagement(connectome, functional_connectivity):
   pass


def node_coupling(struct, func):
   correlation = np.corrcoef(struct, func)

   print(correlation)


node_coupling(connectome, fconn)