import pandas as pd
import numpy as np
import re
import os

PROJECTION = ["AR", 
              "CST", 
              "CBT", 
              "CS", 
              "CT", 
              "F", 
              "OR", 
              "FPT", 
              "OPT", 
              "PPT", 
              "TPT"]

ASSOCIATION = ["AF", 
               "C", 
               "EMC", 
               "FAT", 
               "IFOF", 
               "ILF", 
               "MdLF", 
               "SLF", 
               "U", 
               "UF", 
               "VOF"]

COMMISSURAL = ["AC",
              "CC",
              "PC"
]

CEREBELLUM = [
    "CB",
    "SCP",
    "MCP",
    "ICP",
    "V"
]

BRAINSTEM = [
    "CTT",
    "DLF",
    "LL",
    "ML",
    "MLF",
    "RST",
    "STT"
]

COMPOSITES = ["MEMORY_Composite",
              "LANGUAGE_Composite",
                "EXECUTIVE_Composite",
                "VISUOSPATIAL_Composite",
                "GLOBAL_COGNITIVE_Composite",
                "MMSE"]

TRACT_TYPES = [PROJECTION, ASSOCIATION, COMMISSURAL]
TRACT_NAMES = ["Projection", "Association", "Commissural"]


PATIENT_DATA  = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Belgium Desktop/TAU_Dg_neuro_complet_DATA.csv"
TRACT_DATA ="/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/30_tracts_ordered.csv"

if os.path.exists( "/Users/sam/Desktop/cohens.csv") == False:

    tract_df = pd.read_csv(
        TRACT_DATA, 
        index_col=0
    )
    tract_df["subject"] = tract_df["subject"].str[3:].astype(float)
    tract_df["session"] = tract_df["session"].str.split("-").str[-1].astype(int)


    patient_data = pd.read_csv(
        PATIENT_DATA,
        sep=";", 
        decimal=",", 
        encoding="latin-1"
    )


    merged_df = pd.merge(
        left=patient_data, 
        right=tract_df,
        how="inner",
        left_on=["ID", "Visit_number"],
        right_on=["subject", "session"]
    )


    subset = [col for col in merged_df.columns if col.startswith("mu")]

    df_long = merged_df.melt(
        id_vars=["subject", 
                "Demented", 
                "Diagnostic cognitif détaillé_CLASSIF_1", 
                "tract",
                "MMSE",
                "MEMORY_Composite",
                "LANGUAGE_Composite",
                "EXECUTIVE_Composite",
                "VISUOSPATIAL_Composite",
                "GLOBAL_COGNITIVE_Composite"],
        value_vars=subset,
        var_name="tract_point",
        value_name="value"
    )
    values = ["low", "high"]

    store_dfs = []
    for metric in COMPOSITES:
        df_long["status"] = pd.qcut(
                df_long[metric],
                q=2,
                labels = values
        )
        for i, tract_set in enumerate(TRACT_TYPES):
            # Make a new column, that is Status: high vs low
            #values = ["very low", "low", "high", "very high"]
            #df_long["status"] = np.where(df_long[metric] > med, "high", "low")
            pattern = r"(?:^|_)(?:" + "|".join(map(re.escape, tract_set)) + r")(?:_|$)"

            plot_df = df_long[
                df_long["tract"].str.contains(pattern, na=False, regex=True)
            ].copy()

            means = (
                plot_df.groupby(["tract", "status"])["value"]
                .mean()
                .unstack()
            )
            means["tract_type"] = TRACT_NAMES[i]
            means["metric_"] = metric
            means["mean_diff"] = means["high"] - means["low"]

            means.to_csv(
                f"/Users/sam/Desktop/csvs/{metric}_{TRACT_NAMES[i]}_mean_difs.csv"
            )


            stats = (
                plot_df
                .groupby(["tract", "tract_point", "status"])["value"]
                .agg(["mean", "std", "count"])
                .unstack()
            )
            stats["metric"] = metric
            stats["tract_type"] = TRACT_NAMES[i]
            stats.columns = [f"{stat}_{status}" for stat, status in stats.columns]
            stats["pooled_sd"] = np.sqrt(
                ((stats["count_high"] - 1) * stats["std_high"]**2 +
                (stats["count_low"] - 1) * stats["std_low"]**2) /
                (stats["count_high"] + stats["count_low"] - 2)
            )
            stats["cohens_d"] = (
                (stats["mean_high"] - stats["mean_low"]) / stats["pooled_sd"]
            )
            stats = stats.reset_index()
            store_dfs.append(stats)
            print(stats.columns)

    all_dfs = pd.concat(store_dfs)

    all_dfs.to_csv(
        "/Users/sam/Desktop/cohens.csv"
    )
else:
    all_dfs = pd.read_csv(
        "/Users/sam/Desktop/cohens.csv"
    )

gf = all_dfs.groupby(
    by=["tract", "tract_type_"]
)["cohens_d"].mean().sort_values()

gf.to_csv( "/Users/sam/Desktop/tract_type_cohen_all_metrics.csv")
print(gf)

