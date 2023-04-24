from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

from trustml.exp1.data import get_dataset


class Evaluator(ABC):
    def __init__(self, n_rep: int, n_trials: int) -> None:
        self.dataset = get_dataset().sort_index()
        self.n_rep = n_rep
        self.n_trials = n_trials

        pass

    @classmethod
    def eval_h(cls, res_df: pd.DataFrame, df: pd.DataFrame):
        # res_df = df  # .groupby("arch").max().reset_index()
        results = df.loc[res_df["idxs"]]
        best_perf_inc = results.loc[~results["is_resnet"], "perf"]
        best_perf_res = results.loc[results["is_resnet"], "perf"]

        inc_mu = best_perf_inc.mean()
        inc_max = best_perf_inc.max()
        inc_std = best_perf_inc.std()
        res_std = best_perf_res.std()
        res_mu = best_perf_res.mean()
        res_max = best_perf_res.max()

        _, pvalue = f_oneway(best_perf_inc, best_perf_res)
        # NOTE pvalue is nan when either df are empty.
        return pd.Series(
            dict(
                res_mu=res_mu,
                res_std=res_std,
                pvalue=pvalue,
                inc_mu=inc_mu,
                inc_std=inc_std,
                n_res=best_perf_res.shape[0],
                n_inc=best_perf_inc.shape[0],
                res_max = res_max,
                inc_max = inc_max
            )
        )

    @classmethod
    def eval_fn(cls, arch, df: pd.DataFrame):

        idx = cls.trial_to_idx(arch)
        score = np.random.choice(df.loc[tuple(idx), "perf"]) / 100
        return 1 - score

    @classmethod
    def trial_to_idx(cls, arch: List[int]):

        arch = [int(v) for v in list(arch)]

        return tuple(arch)  # , qlr, qbs)

    @classmethod
    @abstractmethod
    def _run(cls, df: pd.DataFrame, sampler, n_trials):
        raise NotImplementedError()

    def run(self):
        res = []
        for i in range(self.n_rep):
            _res = self._run(self.dataset, n_trials=self.n_trials)
            _res["id"] = i
            _res["sampler"] = type(self).__name__
            dataset = self.dataset
            analysis = self.eval_h(_res, self.dataset)
            res_max_perf = dataset.loc[dataset["is_resnet"], "perf"].max()
            inc_max_perf = dataset.loc[~dataset["is_resnet"], "perf"].max()
            res_perf = dataset.loc[dataset["is_resnet"], "perf"].mean()
            inc_perf = dataset.loc[~dataset["is_resnet"], "perf"].mean()
            res_perf_std = dataset.loc[dataset["is_resnet"], "perf"].std()
            inc_perf_std = dataset.loc[~dataset["is_resnet"], "perf"].std()
            analysis["res_max_diff"] = analysis["res_max"] - res_max_perf
            analysis["inc_max_diff"] = analysis["inc_max"] - inc_max_perf
            analysis["res_diff"] = analysis["res_mu"] - res_perf
            analysis["inc_diff"] = analysis["inc_mu"] - inc_perf
            analysis["res_perf_std_diff"] = analysis["res_std"] - res_perf_std
            analysis["inc_perf_std_diff"] = analysis["inc_std"] - inc_perf_std

            res.append(pd.concat([pd.Series(_res), analysis]))
        df = pd.DataFrame(res)
        df["n_samples"] = self.n_trials
        return df
