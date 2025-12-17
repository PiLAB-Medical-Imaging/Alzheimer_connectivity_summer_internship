"""Script to collect network metrics and build a dataset"""

# Imports
import numpy as np
import pandas as pd
import os
import json
import sys


def retrieve_participant_data(root_direc, subj_id):
    network_types = ["structural", "functional", "combined"]

    participant_data = {}

    # Iterate through and capture participant data. 
    for network in network_types:
        directory = os.path.join(root_direc, subj_id, network, "networks")
        for file_name in os.listdir(directory):
            if file_name.__contains__("json"):
                with open("data.json", "r", encoding="utf-8") as f:
                    data = json.load(f)

                participant_data[f"{network}_{file_name.split(".")[0]}"] = data

    return participant_data

def build_dataset(root_directory, patient_list_json_path):
    with open(patient_list_json_path, "r") as f:
        patient_list = json.load(f)

    for patient in patient_list:
        print(patient)
        patient_data = retrieve_participant_data(root_directory, patient)


if __name__=="__main__":
    root_directory = sys.argv[1]
    patient_list_path = sys.argv[2]
    build_dataset(root_directory, patient_list_path)
    

