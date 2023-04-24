import numpy as np
import pandas as pd
from emukit.core import (
    CategoricalParameter,
    OrdinalEncoding,
    ParameterSpace,
)

from emukit.core.initial_designs.latin_design import LatinDesign

from trustml.exp1.gpy import GPYEvaluator


class Random(GPYEvaluator):
    sampler = None

    @classmethod
    def _run(cls, df: pd.DataFrame, n_trials):
        parameter_space = ParameterSpace(
            [
                CategoricalParameter(f"arch_{i}", OrdinalEncoding(list(range(5))))
                for i in range(6)
            ]
        )

        x = parameter_space.sample_uniform(n_trials)
        result = cls.process_trial(x)

        return {"idxs": result}


class Quasi(GPYEvaluator):
    """
    http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    """

    sampler = None

    @classmethod
    def _run(cls, df: pd.DataFrame, n_trials):
        parameter_space = ParameterSpace(
            [
                CategoricalParameter(f"arch_{i}", OrdinalEncoding(list(range(5))))
                for i in range(6)
            ]
        )

        design = LatinDesign(parameter_space)
        x = design.get_samples(n_trials)
        result = cls.process_trial(x)

        return {"idxs": result}
