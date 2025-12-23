import os
import shutil
import re
import sys


ANATOMICAL_SCAN_TYPE = "T1w"
METADATA_FILES  = ["dataset_description.json", "participants.json", "participants.tsv", "README.md"]

def get_file_extension(filename):
    filename_parts = filename.split(".")

    try:
        if len(filename_parts)  <= 2:
            return "." + filename_parts[1]
        elif len(filename_parts) > 2 :
            return "." + filename_parts[1] + "." + filename_parts[2]
        else:
            return 
    except Exception as e:
        print(f"Errpr occured in the get file extension function: {e}")
        

def extract_session_number(filename):
    """
    Extract a session number from a filename.

    Recognized patterns:
        - 'T#'       → e.g., T1, T2, T3
        - 'ses-#'    → e.g., ses-01, ses-1

    Returns:
        session number as a string (e.g., "1", "01"), or None if not found.
    """
    # Pattern 2: ses-#
    m = re.search(r"ses-(\d+)", filename, re.IGNORECASE)
    if m:
        return m.group(1)
    # Pattern 1: T#
    m = re.search(r"T(\d+)", filename, re.IGNORECASE)
    if m:
        return m.group(1)
    
 

    return None

def extract_participant_number(filename):
    m = re.search(r"(?:sub-)?TAU[_-]?(\d+)", filename, re.IGNORECASE)
    if m:
        if len(m.group(1)) == 1:
             return "00" + m.group(1)
        if len(m.group(1)) == 2:
            return "0" + m.group(1)
        return m.group(1)
    return None

def process_session_num(session_number):
    """
    Adjustable logic to handle strange session number formats as needed/
    
    :param session_number: The session number as a string
    """
    if int(session_number) == 5:
        return "0-5"
    else:
        return session_number
    
def meta_data_creator(destination_folder, meta_data_path):
    """
    This will create the mandatory BIDs metadata folders. Please be
    aware that this will be a blank template. You can adjust the meta_data 
    filepath to point to a completed set of metadata and it will upload that 
    provided you use the correct filenames
    
    :param destination_folder: Filepath pointing to the root directory of your bids file. 
    :param meta_data: A filepath to the folder that contains your metadata. Ensure that the 
    files enclosed within are named dataset_description.json, participants.json, participants.tsv, 
    and README.md
    """
    for file_name in METADATA_FILES:
        path  = os.path.join(meta_data_path, file_name)
        destination = os.path.join(destination_folder, file_name)
        try:
            shutil.copy(path, destination)
        except FileNotFoundError:
            print(f"Please ensure that you have all the required metadata in the folder specified by the meta_data_path\nMissing: {file_name}")
            



def bids_organise(data_folder, destination_folder, data_type, study_name, task="rest", stop_on_failure = False):

    """
    Converts a single folder of scan data into a BIDs organised directory. Currently only works in 
    folders that are pretty homogenous, does not like files that do not contain a subject number for example. 
    
    :param data_folder: Description
    :param destination_folder: Description
    :param data_type: Description
    :param study_name: Description
    :param task: Description
    """
    destination_folder = os.path.join(destination_folder, study_name)
    os.makedirs(destination_folder, exist_ok=True)
    os.makedirs(destination_folder + "/derivatives", exist_ok=True)
   
    os.makedirs(destination_folder, exist_ok=True)

    print(f"Processing: {destination_folder}\n")
    session_set = set()

    count = 0
    for filename in os.listdir(data_folder):
        count = count + 1
        subject_num = extract_participant_number(filename)
        session_num = extract_session_number(filename)

        session_set.add(session_num)

        # Verbose Logging
        print(f"({count})Filename: {filename}")
        print(f"Subject: {subject_num}")
        print(f"Session: {session_num}")



        extension = get_file_extension(filename)

        if data_type == "fMRI" and filename.__contains__(data_type):
            scan_type = "bold"
            func_or_anat = "func"
        elif data_type == "T1" and filename.__contains__("T1"):
            scan_type = "T1w"
            func_or_anat = "anat"
        else:
            print(f"There is no logic for handling a file of this type. The offending file {filename}\n Continuing to the next file.")
            
            if stop_on_failure:
                raise Exception
            else:
                continue
                
        print(f"Scan type: {func_or_anat}")

        # Build the filepath
        try:
            subject_identifier = "TAU" + subject_num
        except TypeError:
            print(f"There is no subject number in this filename: {filename}")
            if stop_on_failure:
                raise Exception
            else: continue

        subject_path = destination_folder + "/" + "sub-" + subject_identifier
        os.makedirs(subject_path, exist_ok=True)
        
        session_path = subject_path + "/" +"ses-" + session_num 
        os.makedirs(session_path, exist_ok=True)

        
        type_path = session_path + "/" + func_or_anat
        os.makedirs(type_path, exist_ok=True)



        if func_or_anat == "func":        
            new_name = "sub-" + subject_identifier + "_ses-" + session_num + "_task-" + task + "_" + scan_type + extension
        elif func_or_anat == "anat":
            new_name = "sub-" + subject_identifier + "_ses-" + session_num + "_" + scan_type + extension
        destination = type_path + "/" + new_name 
        
        print(f"Attempting to save to: {destination}")

        if os.path.exists(destination):
            print("Already exists at location\n")
            continue

        try:
            shutil.copy(data_folder + "/" + filename, destination)
        except FileNotFoundError:
            print(f"Could not find the file: {filename}")
           
            if stop_on_failure:
                print("Exiting")
                raise Exception
            else:
                continue
        
        print("Successfully saved!\n")
    
    print("The values of session number are: ")
    for value in session_set:
        print(value)

    print()


if __name__ == "__main__":
    try:
        data_file_path = sys.argv[1]
        destination_file_path = sys.argv[2]
        data_type = sys.argv[3]
        study_name = sys.argv[4]
        metadata_location = sys.argv[5]
        bids_organise(data_file_path, destination_file_path, data_type, study_name)
        destination_file_path = os.path.join(destination_file_path, study_name)
        meta_data_creator(destination_file_path, metadata_location)
    except IndexError:
        print(IndexError)
        print(f"Invalid arguments. Please enter data filepath (str), destination_filepath (str), data type (str), study name (str), metadata_location (str) \n Received inputs:\ndata: {data_file_path}\ndestination: {destination_file_path}\ntype: {data_type}")
   