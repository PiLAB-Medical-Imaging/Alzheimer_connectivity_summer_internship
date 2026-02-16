import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### Filepaths

NET_DATA = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/OutputData/all_subjects.csv"
PATIENT_DATA  = "/Users/sam/Desktop/TAU_Dg_neuro_complet_DATA.csv"


def merge_dfs(
        clinical_data:str,
        study_data:str
):
    clin_df =  pd.read_csv(
        clinical_data,
        sep=";", 
        decimal=",", 
        encoding="latin-1"
    )
    net_df = pd.read_csv(
        study_data,
        index_col=0
    )

    print(net_df["subj"])
    print(clin_df["ID"])

    combined_df = clin_df.merge(
        net_df,
        how="outer",
        left_on=["ID", "Visit_number"],
        right_on=["subj", "session"]
    )
    return combined_df

def nice_plot(data, metric):
    sns.catplot(
        data=data,
        x="Demented",
        y=metric,
        col="net_type",
        kind="violin",
        sharey=False
    )
    plt.show()

def main():
    merged_data = merge_dfs(
        clinical_data=PATIENT_DATA,
        study_data=NET_DATA
    )

    print(merged_data)

if __name__=="__main__":
    main()