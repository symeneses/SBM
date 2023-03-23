from typing import List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_ess_ps(
        results: pd.DataFrame,
        summaries: List,
        data_sizes: List[int],
        log_scale=False):
    if "ess_mean/s" not in results.columns:
        for k, s in summaries.items():
            results.loc[k, "ess_mean/s"] = s["ess_mean"].min() / \
                results.loc[k, "elapsed_time"]
            results.loc[k, "ess_tail/s"] = s["ess_tail"].min() / \
                results.loc[k, "elapsed_time"]
    else:
        print("Using 'ESS per Second' previously calculated")

    df = results[results["size"].isin(data_sizes)]
    df = pd.melt(df[["library",
                     "sampler",
                     "size",
                     "ess_mean/s",
                     "ess_tail/s"]],
                 id_vars=["library",
                          "size",
                          "sampler"],
                 var_name="metric",
                 value_name="ESS/S")
    df["sampler "] = df[["library", "sampler"]].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)
    df = df.sort_values("ESS/S", ascending=False)

    if len(data_sizes) == 1:
        g = sns.barplot(df,
                        x="sampler ",
                        y="ESS/S",
                        hue="metric",
                        palette="Set2")
        g.set_title(
            f'ESS/S with {data_sizes[0]} rows')
        plt.legend(loc='upper right')
    if len(data_sizes) > 1:
        g = sns.FacetGrid(df, height=5, col="metric")
        g.map_dataframe(sns.lineplot,
                        x="size",
                        y="ESS/S",
                        hue="sampler ",
                        style="library",
                        dashes=False,
                        markers=True,
                        palette="Set2",
                        linewidth=4,
                        alpha=0.5)
        if log_scale == "x":
            g.set(xscale="log")
        if log_scale == "y":
            g.set(yscale="log")
        plt.legend(loc='upper right')
    g.axes[0, 0].tick_params(axis="x", labelrotation=45)
    g.axes[0, 1].tick_params(axis="x", labelrotation=45)
    plt.tight_layout()
    return


def plot_monitor(results: pd.DataFrame):
    results["Peak Memory (MB)"] = results["peak_memory"] / 1e6
    results["Elapsed time (m)"] = results["elapsed_time"] / 60
    df = pd.melt(results[["library",
                          "sampler",
                          "size",
                          "Elapsed time (m)",
                          "Peak Memory (MB)"]],
                 id_vars=["library",
                          "size",
                          "sampler"],
                 var_name="metric",
                 value_name="value")
    df["sampler "] = df[["library", "sampler"]].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)

    g = sns.FacetGrid(df, height=5, col="metric", sharey=False)
    g.map_dataframe(sns.lineplot,
                    x="size",
                    y="value",
                    hue="sampler ",
                    style="library",
                    dashes=False,
                    markers=True,
                    palette="Set2",
                    linewidth=4,
                    alpha=0.5)
    g.add_legend()
    plt.tight_layout()
    return
