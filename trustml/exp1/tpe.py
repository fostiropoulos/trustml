import optuna
import pandas as pd
from optuna import Trial
from optuna.samplers import RandomSampler, TPESampler
from trustml.exp1 import Evaluator


class TPEEvaluator(Evaluator):
    def __init__(self, n_rep: int, n_trials: int):
        super().__init__(n_rep, n_trials)


    @classmethod
    def objective(cls, trial: Trial, df: pd.DataFrame):

        arch = []
        for i in range(6):
            arch.append(str(trial.suggest_categorical(f"arch_{i}", list(range(5)))))
        score = cls.eval_fn(arch, df)
        return score

    @classmethod
    def process_trial(cls, trial: Trial):
        _out = trial.params
        arch = list(_out.values())
        return cls.trial_to_idx(arch)

    @classmethod
    def _run(cls, df: pd.DataFrame, n_trials):

        study = optuna.create_study(sampler=TPESampler())

        study.optimize(lambda x: cls.objective(x, df), n_trials=n_trials)
        trials = study.get_trials()
        idxs = list(map(cls.process_trial, trials))
        return {"idxs": idxs}
