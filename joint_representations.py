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
   SC_file = os.path.join(structural_filepath, f"{subj_id}_sc_matrix.npy")
   FC_file = os.path.join(functional_filepath, f"{subj_id}_fc_matrix.npy")
   SC = np.load(SC_file)
   FC = np.load(FC_file)

   # Run the analyses
   
   # First, run the basic weighting
   simple_weighting_filepath = os.path.join(save_path, f"{subj_id}_simp_weighting.npy")
   if check_existence(simple_weighting_filepath) is False:
      simple_matrix = simple_weighting(SC, FC)
      save_array(simple_weighting_filepath, simple_matrix)






