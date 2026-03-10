import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from analyse_summary_stats import nnf_per_tract
from sklearn.impute import SimpleImputer
from sklearn.decomposition import non_negative_factorization


#TRACT = "mni_edited_PPT_R"
#TRACT = "mni_edited_AC"
#TRACT = "mni_edited_MdLF_R"
TRACT = "mni_edited_FPT_L"
COMPOSITES = [
    "MEMORY_Composite",
    "LANGUAGE_Composite",
    "EXECUTIVE_Composite",
    "VISUOSPATIAL_Composite",
    "GLOBAL_COGNITIVE_Composite",
    "MMSE"
]
COMPONENTS = 3

PATIENT_DATA  = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Belgium Desktop/TAU_Dg_neuro_complet_DATA.csv"
TRACT_DATA ="/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/30_tracts_ordered.csv"
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
             "session",
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
df_long = df_long[df_long["tract"] == TRACT]
df_long["tract_point"] = df_long["tract_point"].str.strip("mu_")
df_long= df_long[df_long["tract_point"]!="31"]

for metric in COMPOSITES:
    df_long["status"] = pd.qcut(
            df_long[metric],
            q=2,
            labels = values
    )

    g = sns.relplot(
        data=df_long,
        x="tract_point",
        y="value",
        hue="status",
        kind="line",
        estimator="mean",
        errorbar=("ci", 95),
        height=4,
        aspect=1.6,      
    )
    g.set_axis_labels("Tract Point", "Mean Value")
    g.set_axis_labels("Tract Point", "Value", fontsize=14)  # x and y labels
    g.set_titles("{col_name}", fontsize=12)  # facet titles
    g._legend.set_title("Status")
    for text in g._legend.get_texts():  # legend labels
        text.set_fontsize(12)
    split_tract_name = TRACT.split("_")
    title = split_tract_name[2]
    for component in split_tract_name[3:]:
        title = title + "_" + component

    for ax in g.axes.flat:
        ax.set_xticks(np.arange(0, 30, 5))  # ticks every 20

    #plt.title(f"{title} for {metric}")
    whole_title = f"{title}_{metric}"

    #plt.show()
    path = os.path.join(
        "/Users/sam/Desktop/",
        whole_title +".png"
    )
    g.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    nn_df = nnf_per_tract(
        merged_df=merged_df,
        tract=TRACT,
        components=COMPONENTS,
        metric=metric
    )

    g2 = sns.catplot(
        data=nn_df,
        x="status",
        y="value",
        hue="status",
        col="component_name",
        kind="box",
        col_wrap=1,
        height=3,
        aspect=2,
        sharey=False
    )

    for ax, comp in zip(g2.axes.flat, nn_df["component_name"].unique()):
        sns.stripplot(
            data=nn_df[nn_df["component_name"] == comp],
            x="status",
            y="value",
            hue="Demented",
            ax=ax,
            size=5,
            alpha=0.5
        )
    g2.savefig(
         f"/Users/sam/Desktop/component_{metric}.png",
         bbox_inches="tight",
         dpi=600
    )
    plt.close() 
    #plt.show()




    