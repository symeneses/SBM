import random
from typing import Dict

import arviz as az


def dist_validator(infer_data: Dict, seed: int):
    random.seed(seed)
    ref_key = random.sample(list(infer_data.keys()), 1)[0]
    summaries = {k: az.summary(d, stat_funcs={
        "ess_mean": lambda x: az.ess(x, method="mean")})
        for k, d in infer_data.items()}
    ref_summary = summaries[ref_key]

    results = {}
    print("Percentage of variables with the expected mean using MCSE:")
    min_mean, max_mean = ref_summary["mean"] - 3 * ref_summary["mcse_mean"], \
        ref_summary["mean"] + 3 * ref_summary["mcse_mean"]
    for k, s in summaries.items():
        s["min_mean"], s["max_mean"] = min_mean, max_mean
        s["test"] = s.apply(lambda x: x["min_mean"] <=
                            x["mean"] <= x["max_mean"], axis=1)
        results[k] = s["test"].mean()
        indicator = "\u2705" if s["test"].mean() >= 0.95 else "\u274C"
        print((f"> {k:<16}: {s['test'].mean():>7.2%} using {ref_key}"),
              (f"as a random reference {indicator}"))

    return results, summaries
