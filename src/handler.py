# Handle sampling for all models
import arviz as az
import pandas as pd

from src.sampler import Sampler

OUTPUT_PATH = "../data/results"


class Handler:
    def __init__(self, models, datasets, pymc_samplers) -> None:
        self.models = models
        self.datasets = datasets
        self.pymc_samplers = pymc_samplers

    def execute(
            self,
            draws: int,
            tune: int,
            chains: int,
            seed: int) -> az.InferenceData:
        sampler = Sampler(self.datasets, self.pymc_samplers)
        results = pd.DataFrame()
        infer_data = {}
        for n, m in self.models.items():
            print(f"\n> Getting samples using libray {n}:")
            samples = sampler.fit(m, draws, tune, chains, seed)
            for s in samples:
                key = f"{n}_{s[0]}_{s[1]}"
                results.loc[key, "library"] = n
                results.loc[key, "sampler"] = s[0]
                results.loc[key, "size"] = int(s[1])
                results.loc[key, s[3].keys()] = s[3].values()
                infer_data[key] = s[2]
                s[2].to_netcdf(f"{OUTPUT_PATH}/{key}.nc")
        results.to_pickle(f"{OUTPUT_PATH}/results.pkl")
        return infer_data, results
