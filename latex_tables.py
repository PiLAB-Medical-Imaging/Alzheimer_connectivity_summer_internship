import pandas as pd


df = pd.read_csv("/Users/sam/Desktop/correlation_data.csv", header=2)
df.columns = ['network', 'measure', 'outcome', 'funct_p', 'struct_p', 'sw_p', "funct_r", "struct_r", "sw_r"]
df['outcome'] = df['outcome'].str.replace("_", " ")


def star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

for t in ["funct","struct","sw"]:
    df[f"{t}_fmt"] = df[f"{t}_r"].round(2).astype(str) + df[f"{t}_p"].apply(star)


def make_rows(df_net):
    rows = []

    for measure in df_net['measure'].unique():
        sub = df_net[df_net['measure'] == measure]
        first = True
        for _, row in sub.iterrows():
            # Use multirow for the first row of each measure
            if first:
                rows.append(
                    f"\\multirow{{{len(sub)}}}{{*}}{{{measure.replace('_',' ').title()}}} & "
                    f"{row['outcome']} & {row['funct_fmt']} & {row['struct_fmt']} & {row['sw_fmt']} \\\\"
                )
                first = False
            else:
                rows.append(
                    f"& {row['outcome']} & {row['funct_fmt']} & {row['struct_fmt']} & {row['sw_fmt']} \\\\"
                )
        rows.append("\\midrule")
    # Remove the last midrule
    return "\n".join(rows[:-1])

def make_table(df, network_name):
    df_net = df[df['network'] == network_name]
    rows = make_rows(df_net)

    latex = f"""
\\begin{{table}}[htbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Correlations between network measures and cognitive outcomes for the {network_name} network.}}
\\small
\\begin{{tabular}}{{llccc}}
\\toprule
Measure & Outcome & Functional & Structural & SWFC \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\footnotesize
\\item Values represent Pearson correlation coefficients ($r$).
\\item * $p<0.05$, ** $p<0.01$, *** $p<0.001$.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""
    return latex

networks = df['network'].unique()
for net in networks:
    latex_code = make_table(df, net)
    print(latex_code)  # Or save to a file

with open("network_tables.tex", "w") as f:
    for net in networks:
        f.write(make_table(df, net))
        f.write("\n\n")