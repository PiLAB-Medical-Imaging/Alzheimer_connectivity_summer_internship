from os import listdir
import sys
from os.path import join
import pandas as pd

def subject_wise_compilation(directory, save_name):
    all_dfs = []
    for file in listdir(directory):
        print(file.split(".")[-1])
        if file.split(".")[-1] == "csv":
            file_path = join(
                directory,
                file
            )
            try:
                df = pd.read_csv(
                    filepath_or_buffer=file_path,
                    index_col=0
                )
                all_dfs.append(df)
            except Exception as e:
                print(file)
                continue

    final_df = pd.concat(all_dfs)

    save_path = join(
        directory, 
        save_name
    )

    final_df.to_csv(save_path)


if __name__ == "__main__":
    directory = sys.argv[1]
    save_name = sys.argv[2]
    subject_wise_compilation(
        directory=directory,
        save_name=save_name
    )