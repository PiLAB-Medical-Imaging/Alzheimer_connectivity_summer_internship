import numpy as np
import pandas as pd
import os
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sbs
import chardet

NETWORK_DATA = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/compiled_data.csv"
PATIENT_DATA = "/Users/sam/Desktop/TAU_Dg_neuro_complet_DATA.csv" 

net_data = pd.read_csv(
    filepath_or_buffer=NETWORK_DATA,
    index_col=0
)
patient_data = pd.read_csv(
    filepath_or_buffer=PATIENT_DATA,
    sep=";", 
    decimal=",", 
    encoding="latin-1"
)

print(patient_data.head())
print(net_data.head())
pd.pivot()