from os import listdir
import sys
from os.path import join
import pandas as pd

def subject_wise_compilation(directory):
    all_dfs = []
    for file in listdir(directory):
        file_path = join(
            directory,
            file
        )
        df = pd.read_csv(
            filepath_or_buffer=file_path,
            index_col=0
        )
        all_dfs.append(df)

    final_df = pd.concat(all_dfs)
    return final_df


if __name__ == "__main__":
    directory = sys.argv[1]
    subject_wise_compilation(directory=directory)