# Fit the model depending on the library, processor, and backend
import os
from enum import Enum
from inspect import getfullargspec
from typing import Dict, List, Tuple

import arviz as az
import pandas as pd
import pymc as pm
from jax import random
from numpyro import infer
from pymc.sampling import jax

from src.utils import monitor


class ModelType(Enum):
    PYMC = 1
    NUMPYRO = 2


PYMC_SAMPLERS = {
    "default": pm.sample,
    "numpyro": jax.sample_numpyro_nuts,
    "blackjax": jax.sample_blackjax_nuts

}


class Sampler:
    def __init__(self,
                 datasets: Dict[str,
                                pd.DataFrame],
                 pymc_samplers: List[str]) -> None:
        self.datasets = datasets
        self.pymc_samplers = pymc_samplers

    def fit(self, generator, draws: int, tune: int, chains: int, seed: int,
            model_args: Dict = {}) -> Tuple[List[az.InferenceData], Dict]:
        results = []
        type_ = Sampler._get_type(generator)
        if type_ == 1:
            for k, d in self.datasets.items():
                print(f"\n>> Getting samples for data with size {k}")
                model = generator(d)
                for s in self.pymc_samplers:
                    print(f"\n>>> Getting samples using the PYMC sampler {s}")
                    with model:
                        data, metrics = sampling_pymc(s,
                                                      draws=draws,
                                                      tune=tune,
                                                      chains=chains,
                                                      seed=seed)
                        results.append((s, k, data, metrics))
            return results
        if type_ == 2:
            args = getfullargspec(generator.model).args
            for k, d in self.datasets.items():
                print(f"\n>> Getting samples for data with size {k}")
                model_args = {a: d[a].values for a in args}
                data, metrics = sampling_numpyro(generator, draws, tune,
                                                 chains, model_args)
                results.append(("default", k, data, metrics))
            return results

    def _get_type(generator) -> ModelType:
        if callable(generator) and getfullargspec(
                generator).annotations["return"] == pm.model.Model:
            return 1
        if isinstance(generator, infer.hmc.NUTS):
            return 2


@monitor
def sampling_pymc(sampler, draws: int, tune: int,
                  chains: int, seed: int) -> az.InferenceData:
    sampler_pymc = PYMC_SAMPLERS[sampler]
    extra_args = {}
    if 'cores' in getfullargspec(sampler_pymc).args:
        extra_args['cores'] = min(os.cpu_count(), chains)
    if 'progressbar' in getfullargspec(sampler_pymc).args:
        extra_args['progressbar'] = False
    data = PYMC_SAMPLERS[sampler](
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=seed,
        **extra_args,
        idata_kwargs={
            'log_likelihood': False})
    return data


@monitor
def sampling_numpyro(sampler, draws: int, tune: int,
                     chains: int, model_args) -> az.InferenceData:
    rng_key = random.PRNGKey(0)
    mcmc = infer.MCMC(
        sampler,
        num_warmup=tune,
        num_samples=draws,
        num_chains=chains)
    mcmc.run(
        rng_key,
        **model_args)

    data = az.from_numpyro(mcmc, log_likelihood=False)
    return data
