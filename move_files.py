import os
import sys
from os.path import join, exists
import shutil

ENGAGEMENT_NAME = "engagement.nii.gz"

def move_files(
        outputs_folder,
        destination, 
        file_name
):
    os.makedirs(
        name=destination, 
        exist_ok=True
    )
    for subject in os.listdir(outputs_folder):
        print(f'Analysing {subject}')
        subject_folder = join(
            outputs_folder,
            subject
        )
        for session in os.listdir(subject_folder):
            print(f"\t{session}")
            session_folder = join(
                subject_folder,
                session
            )
            eng_path = join(
                session_folder,
                file_name
            )
            new_name = join(
                destination, 
                subject + "_" + session + "_" + file_name
            )
            if exists(eng_path):
                shutil.copyfile(
                    src=eng_path,
                    dst=new_name
                )

if __name__ == "__main__":
    outputs_folder = sys.argv[1]
    destination = sys.argv[2]
    print("goose")
    move_files(
        outputs_folder=outputs_folder,
        destination=destination, 
        file_name=ENGAGEMENT_NAME
    )