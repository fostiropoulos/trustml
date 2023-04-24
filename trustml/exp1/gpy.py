from abc import abstractmethod
import numpy as np
import pandas as pd
from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core import (
    CategoricalParameter,
    OrdinalEncoding,
    ParameterSpace,
)
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.model_wrappers import GPyModelWrapper
from GPy.models import GPRegression

from trustml.exp1 import Evaluator


class GPYEvaluator(Evaluator):
    def __init__(self, n_rep: int, n_trials: int):
        super().__init__(n_rep, n_trials)

    @classmethod
    @abstractmethod
    def sampler(cls, model):
        raise NotImplementedError

    @classmethod
    def objective(cls, xs: pd.DataFrame, df: pd.DataFrame):

        return np.array([cls.eval_fn(x - 1, df) for x in xs])[:, None]

    @classmethod
    def process_trial(cls, xs):
        return [(np.array(cls.trial_to_idx(x)) - 1).tolist() for x in xs]

    @classmethod
    def _run(cls, df: pd.DataFrame, n_trials):

        parameter_space = ParameterSpace(
            [
                CategoricalParameter(f"arch_{i}", OrdinalEncoding(list(range(5))))
                for i in range(6)
            ]
        )

        num_data_points = 1
        x = parameter_space.sample_uniform(num_data_points)

        def objective(xs):
            return cls.objective(xs, df)

        y = objective(x)

        model_gpy = GPRegression(x, y)
        model_emukit = GPyModelWrapper(model_gpy)

        bayesopt_loop = BayesianOptimizationLoop(
            model=model_emukit,
            space=parameter_space,
            acquisition=cls.sampler(model_emukit),
            batch_size=1,
        )
        max_iterations = n_trials - num_data_points
        bayesopt_loop.run_loop(objective, max_iterations)

        result = cls.process_trial(model_emukit.X)
        return {"idxs": result}


class Greedy(GPYEvaluator):
    sampler = lambda model: NegativeLowerConfidenceBound(model, beta=0)


class CombinedVariance(GPYEvaluator):
    sampler = lambda model: NegativeLowerConfidenceBound(model, beta=1)


class Variance(GPYEvaluator):
    sampler = ModelVariance
