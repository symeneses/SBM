# Handle sampling for all models
import arviz as az
import pandas as pd

from src.sampler import Sampler

OUTPUT_PATH = "../data/results"


class Handler:
    def __init__(self, models, data, pymc_samplers) -> None:
        self.models = models
        self.data = data
        self.pymc_samplers = pymc_samplers

    def execute(self, draws: int, tune: int, chains: int, seed: int) -> az.InferenceData:
        sampler = Sampler(self.data, self.pymc_samplers)
        results = pd.DataFrame()
        infer_data = {}
        for n, m in self.models.items():
            print(f"Getting samples using libray {n}:\n")
            samples = sampler.fit(m, draws, tune, chains, seed)
            for s in samples:
                results.loc[f"{n}_{s[0]}", "library"] = n
                results.loc[f"{n}_{s[0]}", "sampler"] = s[0]
                results.loc[f"{n}_{s[0]}", s[2].keys()] = s[2].values()
                infer_data[f"{n}_{s[0]}"] = s[1]
                s[1].to_netcdf(f"{OUTPUT_PATH}/{n}_{s[0]}.nc")
        results.to_pickle(f"{OUTPUT_PATH}/results.pkl")
        return infer_data, results
