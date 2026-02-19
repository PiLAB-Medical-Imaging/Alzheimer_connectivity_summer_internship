import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import non_negative_factorization
from sklearn.impute import SimpleImputer
### Filepaths

NET_DATA = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/OutputData/all_subjects.csv"
PATIENT_DATA  = "/Users/sam/Desktop/TAU_Dg_neuro_complet_DATA.csv"
TRACT_DATA ="/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/all_tracts_30.csv"
COMPOSITES = ["MEMORY_Composite",
                "EXECUTIVE_Composite",
                "VISUOSPATIAL_Composite",
                "GLOBAL_COGNITIVE_Composite"]

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

TRACT_TYPES = [PROJECTION, ASSOCIATION, COMMISSURAL]


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


def analyse_tract_engagement(
        tract_data = TRACT_DATA,
        original_data = PATIENT_DATA
):
    tract_df = pd.read_csv(
        tract_data, 
        index_col=0
    )
    patient_data = pd.read_csv(
        original_data,
        sep=";", 
        decimal=",", 
        encoding="latin-1"
    )
    tract_df["subject"] = tract_df["subject"].str[3:].astype(float)
    tract_df["session"] = tract_df["session"].str.split("-").str[-1].astype(int)
    print(tract_df.head())
    # Merge:
    merged_df = pd.merge(
        left=patient_data, 
        right=tract_df,
        how="inner",
        left_on=["ID", "Visit_number"],
        right_on=["subject", "session"]
    )

    #plot_along_tracts(merged_df)
    #seaborn_tracts(merged_df=merged_df, type="sigma")

    for metric in COMPOSITES:
        plot_high_low(
            merged_df=merged_df,
            metric=metric
        )
    
    """for tracts in TRACT_TYPES:
        mean_plots(
            merged_df=merged_df,
            relevant_set=tracts,
            name=tracts
        )"""

    """tracts_scores(
        merged_df,
        "mni_edited_CT_L",
        "MMSE"
    )
    
    tract_cols = [f"mu_{i}" for i in range(1, 101)]
    plot_tract_behavior(
        df=merged_df,
        tract_name="mni_edited_C_L",
        tract_cols=tract_cols,
        cog_score="MMSE"
    )
    """

def plot_along_tracts(merged_df:pd.DataFrame):
    subset = [col for col in merged_df.columns if col.startswith("mu")]
    grouped = (merged_df.groupby(["Demented", "tract"])[subset]
               .mean()
               .reset_index()
               )
    
    mean_df  = grouped[subset].mean()
    std_df   = grouped[subset].std()
    count_df = grouped[subset].count()
    sem_df = std_df / np.sqrt(count_df)
    ci_upper = mean_df + 1.96 * sem_df
    ci_lower = mean_df - 1.96 * sem_df
    for tract_name, tract_df in grouped.groupby("tract"):
    
        plt.figure()
        
        for dementia_status, group_df in tract_df.groupby("Demented"):
            y = group_df[subset].values.flatten()
            x = range(1, len(subset) + 1)
            
            plt.plot(x, y, label=f"Demented = {dementia_status}")
        
        plt.title(f"Mean Tract Profile - {tract_name}")
        plt.xlabel("Tract Point")
        plt.ylabel("Mean Value")
        plt.legend()
        plt.show()

def seaborn_tracts(merged_df:pd.DataFrame, type = "mu"):
    subset = [col for col in merged_df.columns if col.startswith(type)]
    # Convert to numpy array
    data_array = merged_df[subset].values  # shape (num_rows, num_points)

    # Apply smoothing along tract points (columns)
    smoothed_array = gaussian_filter1d(data_array, sigma=1, axis=1)
    smoothed_cols = [f"smoothed_{col}" for col in subset]

    merged_df[smoothed_cols] = smoothed_array
    df_long = merged_df.melt(
        id_vars=["subject", "Demented", "tract"],
        value_vars=smoothed_cols,
        var_name="tract_point",
        value_name="value"
    )
    df_long["tract_point"] = (
        df_long["tract_point"]
        .str.replace(f"smoothed_{type}_", "")
        .astype(int)
    )
    print(df_long.head())
    sns.set(style="whitegrid")

    g = sns.relplot(
        data=df_long,
        x="tract_point",
        y="value",
        hue="Demented",
        col="tract",
        kind="line",
        estimator="mean",
        errorbar=("ci", 95),
        height=4,
        aspect=1.2,
        col_wrap=8,
        facet_kws={"sharey": False},
        
    )

    g.set_axis_labels("Tract Point", "Mean Value")
    plt.show()
    

    g = sns.relplot(
        data=df_long,
        x="tract_point",
        y="value",
        hue="subject",
        col="tract",
        col_wrap=8,
        kind="line",
        estimator=None,
        units="subject",
        alpha=0.25,
        linewidth=1,
        legend=False,
        facet_kws={"sharey": False}
    )

    g.set_axis_labels("Tract Point", "Value")
    plt.show()



def tracts_scores(
        merged_df: pd.DataFrame, 
        tract_name, 
        cog_score
):
    mu_cols = [col for col in merged_df.columns if col.startswith("mu_")]
    tract_df = merged_df[merged_df["tract"] == tract_name].copy()
    tract_df = tract_df[tract_df["Demented"] == 1.0].copy()
    sns.set(style="whitegrid")

    for mu in mu_cols:
        plt.figure(figsize=(5,4))
        
        # Scatter plot with regression line
        sns.regplot(
            data=tract_df,
            x=mu,
            y=cog_score,
            scatter_kws={"alpha":0.6},
            line_kws={"color":"red"},
        )
        plt.xlabel(f"{mu} value")
        plt.ylabel(cog_score)
        r, p = pearsonr(tract_df[mu], tract_df[cog_score])
        plt.title(f"{tract_name} — {mu} vs {cog_score} (r={r:.2f}, p={p:.3f})")
        plt.tight_layout()
        plt.show()


def plot_tract_behavior(
    df,
    tract_name,
    tract_cols,
    cog_score,
    hue_col=None,
    points_per_row=5,
    lowess=False,
    figsize_per_plot=(4,3),
    save_path=None
):
    """
    Plot scatter + regression line between tract points and a continuous cognitive score.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing tract values and cognitive scores.
    tract_name : str
        Name of the tract to select in the 'tract' column.
    tract_cols : list of str
        Column names of tract points (e.g., ['mu_1', 'mu_2', ...]).
    cog_score : str
        Name of the continuous cognitive score (e.g., 'MMSE').
    hue_col : str, optional
        Column name for categorical grouping (e.g., 'Demented').
    points_per_row : int
        Number of tract points per row of subplots.
    lowess : bool
        If True, fit lowess curve instead of linear regression.
    figsize_per_plot : tuple
        Size of each subplot (width, height).
    save_path : str or None
        Path to save the figure. If None, figure is just shown.
    """
    
    # Filter tract
    tract_df = df[df["tract"] == tract_name].copy()
    
    num_points = len(tract_cols)
    num_rows = int(np.ceil(num_points / points_per_row))
    
    fig, axes = plt.subplots(
        num_rows, points_per_row, 
        figsize=(points_per_row*figsize_per_plot[0], num_rows*figsize_per_plot[1])
    )
    axes = axes.flatten()
    
    sns.set(style="whitegrid")
    
    for i, mu in enumerate(tract_cols):
        ax = axes[i]
        
        if hue_col:
            # Scatter with hue
            sns.scatterplot(
                data=tract_df,
                x=mu,
                y=cog_score,
                hue=hue_col,
                alpha=0.6,
                ax=ax,
                legend=False
            )
            sns.regplot(
                data=tract_df,
                x=mu,
                y=cog_score,
                scatter=False,
                lowess=lowess,
                ax=ax,
                line_kws={"color":"red"}
            )
        else:
            sns.regplot(
                data=tract_df,
                x=mu,
                y=cog_score,
                scatter_kws={"alpha":0.6, "s":30},
                line_kws={"color":"red"},
                lowess=lowess,
                ax=ax
            )
        
        # Pearson r
        r, _ = pearsonr(tract_df[mu], tract_df[cog_score])
        ax.set_title(f"{mu} (r={r:.2f})")
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    
    # Common axis labels
    fig.text(0.5, 0.04, f"{tract_name} tract value", ha='center', fontsize=12)
    fig.text(0.04, 0.5, cog_score, va='center', rotation='vertical', fontsize=12)
    fig.suptitle(f"{tract_name} — {cog_score} vs Tract Points", fontsize=14)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()




def plot_high_low(
        merged_df: pd.DataFrame,
        metric:str
):
    """
   Split the dataset according to participants who are
   either high or low on a given metric
    
    :param merged_df: Description
    :type merged_df: pd.DataFrame
    :param metric: Description
    :type metric: str
    """
    subset = [col for col in merged_df.columns if col.startswith("mu")]

    df_long = merged_df.melt(
        id_vars=["subject", 
                 "Demented", 
                 "Diagnostic cognitif détaillé_CLASSIF_1", 
                 "tract",
                 "MEMORY_Composite",
                 "EXECUTIVE_Composite",
                 "VISUOSPATIAL_Composite",
                 "GLOBAL_COGNITIVE_Composite"],
        value_vars=subset,
        var_name="tract_point",
        value_name="value"
    )

    # Make a new column, that is Status: high vs low
    med = np.nanmedian(df_long[metric])
    #values = ["very low", "low", "high", "very high"]
    #df_long["status"] = np.where(df_long[metric] > med, "high", "low")
    values = ["low", "high"]
    
    df_long["status"] = pd.qcut(
        df_long[metric],
        q=2,
        labels = values
    )
    sns.set_theme(style="whitegrid")

    g = sns.relplot(
        data=df_long,
        x="tract_point",
        y="value",
        hue="status",
        col="tract",
        kind="line",
        estimator="mean",
        errorbar=("ci", 95),
        height=4,
        aspect=1.2,
        col_wrap=8,
        facet_kws={"sharey": False},
        
    )

    g.set_axis_labels("Tract Point", "Mean Value")
    plt.show()

    
def mean_plots(
        merged_df: pd.DataFrame,
        relevant_set: list,
        name
):
    
    subset = [col for col in merged_df.columns if col.startswith("mu")]

    df_long = merged_df.melt(
        id_vars=["subject", 
                 "Demented", 
                 "Diagnostic cognitif détaillé_CLASSIF_1", 
                 "tract",
                 "CLASSIF_REVUE_categ",
                 "MEMORY_Composite",
                 "EXECUTIVE_Composite",
                 "VISUOSPATIAL_Composite",
                 "GLOBAL_COGNITIVE_Composite"],
        value_vars=subset,
        var_name="tract_point",
        value_name="value"
    )
    print(df_long["tract"].unique())
    print(len(df_long["tract"].unique()))
    df_long = df_long[
        df_long["tract"].str.contains("|".join(relevant_set),na=False)
    ]    

    print(df_long["tract"].unique())
    print(len(df_long["tract"].unique()))

    print(name)
    sns.catplot(
        data=df_long,
        x="CLASSIF_REVUE_categ",
        y="value",
        col="tract",
        col_wrap=8,
        hue="CLASSIF_REVUE_categ",
        sharex=True,
        sharey=False,
        kind="violin"
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
        facet_kws={"sharey": False},
        col_wrap=4
    )
    plt.show()


def nnf_per_tract(
        merged_df,
        tract,
        components = 3
):
    cols = [col for col in merged_df.columns if col.startswith("mu")]  
    imputer = SimpleImputer(strategy="median")
    tract_df = merged_df[merged_df["tract"] == tract].copy()
    tract_data = tract_df[cols]
    tract_data = imputer.fit_transform(tract_data)
    # Note that H is the component matrix and W is the weights
    #  matrix (transformed data)
    W, H, n_iter = non_negative_factorization(
        X=tract_data,
        n_components=components,
        init="random",
        random_state=0,
        max_iter=10000,
    )

    component_cols = [f"C_{i+1}" for i in range(W.shape[1])]
    W_df = pd.DataFrame(
        W,
        columns=component_cols,
        index=tract_df.index
    )

    result_df = pd.concat(
        [
            tract_df[["subject", 
                 "Demented", 
                 "Diagnostic cognitif détaillé_CLASSIF_1", 
                 "tract",
                 "CLASSIF_REVUE_categ",
                 "MEMORY_Composite",
                 "EXECUTIVE_Composite",
                 "VISUOSPATIAL_Composite",
                 "GLOBAL_COGNITIVE_Composite"]],
            W_df
        ],
        axis=1
    )

    print(W.shape)
    print(H.shape)

    for i in range(components):
        plt.plot(H[i,:])
    plt.show()

    # Longify the result_df
    df_long = pd.melt(
        result_df,
        id_vars=["subject", 
                 "Demented", 
                 "Diagnostic cognitif détaillé_CLASSIF_1", 
                 "tract",
                 "CLASSIF_REVUE_categ",
                 "MEMORY_Composite",
                 "EXECUTIVE_Composite",
                 "VISUOSPATIAL_Composite",
                 "GLOBAL_COGNITIVE_Composite"],
        value_vars=component_cols,
        var_name="component_name",
        value_name="value"
    )

    #values = ["very low", "low", "high", "very high"]
    values = ["LOW", "HIGH"]
    #df_long["status"] = np.where(df_long[metric] > med, "high", "low")
    df_long["status"] = pd.qcut(
        df_long["MEMORY_Composite"],
        q=2,
        labels = values
    )

    sns.set_theme(style="whitegrid")

    g = sns.relplot(
        data=df_long,
        x="component_name",
        y="value",
        hue="status",
        kind="line",
        estimator="mean",
        errorbar=("ci", 95),
        height=4,
        aspect=1.2,
        facet_kws={"sharey": False},
        
    )

    g.set_axis_labels("Tract Point", "Mean Value")  
    plt.title(f"Component Analysis: {tract}")
    plt.show()
    


def temp(tract_data, original_data):
    tract_df = pd.read_csv(
        tract_data, 
        index_col=0
    )
    patient_data = pd.read_csv(
        original_data,
        sep=";", 
        decimal=",", 
        encoding="latin-1"
    )
    tract_df["subject"] = tract_df["subject"].str[3:].astype(float)
    tract_df["session"] = tract_df["session"].str.split("-").str[-1].astype(int)
    print(tract_df.head())
    # Merge:
    merged_df = pd.merge(
        left=patient_data, 
        right=tract_df,
        how="inner",
        left_on=["ID", "Visit_number"],
        right_on=["subject", "session"]
    )
    for tract in merged_df["tract"].unique():
        nnf_per_tract(
            merged_df=merged_df,
            tract=tract
        )

def main():
    merged_data = merge_dfs(
        clinical_data=PATIENT_DATA,
        study_data=NET_DATA
    )

    print(merged_data)
    #rel_plots(long_data=longer_df, variable="global_clustering")

if __name__=="__main__":
    #analyse_tract_engagement()
    temp(TRACT_DATA,PATIENT_DATA)
