from typing import List

import pandas as pd
import seaborn as sns


def plot_ess_ps(results: pd.DataFrame, summaries: List):
    if "ess_mean/s" not in results.columns:
        for k, s in summaries.items():
            results.loc[k, "ess_mean/s"] = s["ess_mean"].min() / \
                results.loc[k, "elapsed_time"]
            results.loc[k, "ess_tail/s"] = s["ess_tail"].min() / \
                results.loc[k, "elapsed_time"]
    else:
        print("Using 'ESS per Second' previously calculated")

    df = pd.melt(results[["library",
                          "sampler",
                          "ess_mean/s",
                          "ess_tail/s"]],
                 id_vars=["library",
                          "sampler"],
                 var_name="metric",
                 value_name="ESS/S")
    df["sampler"] = df[["library", "sampler"]].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)
    sns.barplot(
        data=df,
        x="sampler",
        y="ESS/S",
        hue="metric",
        palette="pastel")
    return
