import networkx as nx
import numpy as np
import os


############### Network Building Functions ###########

def engagement(connectome, functional_connectivity):
   """ To implement: Engagement Metrics"""
   pass


def node_coupling(struct, func):
   pass


def simple_weighting(SC, FC):
   """Simple element wise multiplcation of structural and functional connectivity"""

   # Start with a simple normalised structural connectivity matrix each entry is normalised to a 0-1 value, with 1 corresponding to the strongest connection
   normed_SC = SC/np.max(SC)

   # Simple element wise weighting
   resultant = np.multiply(normed_SC, FC)

   return resultant


################## Utility Functions ###############
def save_array(file_name, array, verbose=True):
   np.save(array, file_name)
   if verbose:
      print(f"Saving {file_name}.")

def check_existence(filepath):
   if os.path.exists(filepath):
      return True
   return False


############### Driver Function ###################
def create_combined_matrices(structural_filepath, functional_filepath, subj_id, save_path, overwrite = False):

   # Load the raw matrices
   SC = np.load(structural_filepath)
   FC = np.load(functional_filepath)

   # Run the analyses
   
   # First, run the basic weighting
   simple_matrix = simple_weighting(SC, FC)
   return simple_matrix
      






