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
    net_df["subj"] = net_df["subj"].str.split("_").str[0].astype(float)

    combined_df = clin_df.merge(
        net_df,
        how="inner",
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

def longify_data(data:pd.DataFrame):
    metric_columns = []
    prefixes = ("emot_", "dmn", "salience", "ecn", "m_connecivity", "diamter", "global_clustering", "isolates", "density")
    for col in data.columns:
        if col.startswith(prefixes):
            metric_columns.append(col)

    df_long = data.melt(
        id_vars=[col for col in data.columns if col not in metric_columns],
        value_vars=metric_columns,
        var_name="metric",
        value_name="score"
    )

    return df_long

def category_plots(long_data, variable):
    plotting_data = long_data[long_data["metric"] == variable]
    sns.catplot(
        plotting_data, 
        x = "Diagnostic cognitif détaillé_CLASSIF_1",
        y = "score", 
        hue= "Diagnostic cognitif détaillé_CLASSIF_1",
        col="net_type",
        kind="violin",
        sharey=False
    )
    plt.show()

def rel_plots(long_data, variable):
    
    plotting_data = long_data[long_data["metric"] == variable]
    sns.relplot(
        plotting_data, 
        x="score",
        y="MMSE",
        col="net_type",
        hue="Diagnostic cognitif détaillé_CLASSIF_1",
        facet_kws={"sharey": False}
    )
    plt.show()

def main():
    merged_data = merge_dfs(
        clinical_data=PATIENT_DATA,
        study_data=NET_DATA
    )

    longer_df = longify_data(merged_data)

    category_plots(long_data=longer_df, variable= "salience_density")
    #rel_plots(long_data=longer_df, variable="global_clustering")

if __name__=="__main__":
    main()