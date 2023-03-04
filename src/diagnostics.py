import random
import warnings
from typing import Dict

import arviz as az
import numpy as np


def dist_validator(infer_data: Dict, seed: int = None, ref_key: str = None):
    if seed:
        random.seed(seed)
        ref_key = random.sample(list(infer_data.keys()), 1)[0]
    ref_key_no_size = "_".join(ref_key.split("_")[:-1])
    summaries = {k: az.summary(d, extend=False, stat_funcs={
        "ess_mean": lambda x: az.ess(x, method="mean"),
        "ess_tail": lambda x: az.ess(x, method="tail"),
        "mean": np.mean,
        "mcse_mean": lambda x: az.mcse(x, method="mean")}
    ) for k, d in infer_data.items()}

    results = {}
    print("Percentage of variables with the expected mean using MCSE:")
    for k, s in summaries.items():
        s_size = k.split("_")[-1]
        ref_key = f"{ref_key_no_size}_{s_size}"
        ref_summary = summaries[ref_key]
        min_mean, max_mean = \
            ref_summary["mean"] - 3 * ref_summary["mcse_mean"], \
            ref_summary["mean"] + 3 * ref_summary["mcse_mean"]
        s["min_mean"], s["max_mean"] = min_mean, max_mean
        s["test"] = s.apply(lambda x: x["min_mean"] <=
                            x["mean"] <= x["max_mean"], axis=1)
        results[k] = s["test"].mean()
        indicator = "\u2705" if s["test"].mean() >= 0.95 else "\u274C"
        print((f"> {k:<24}: {s['test'].mean():>7.2%} using as a"),
              (f"{'random ' if seed else ''}reference"),
              (f"{ref_key :<24} {indicator}"))

    return results, summaries


def convergency_validator(infer_data: Dict):
    print("Convergency test using R-hat:")

    with warnings.catch_warnings():
        for k, d in infer_data.items():
            warnings.simplefilter("ignore")
            rhat = max(az.rhat(d).max().values()).values
            indicator = "\u2705" if rhat < 1.05 else "\u274C"
            print(f"> {k:<24}: {rhat:>2.2f} {indicator}")
    return
