from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
from trustml import data_dir, results_dir
from trustml.exp1.data import get_dataset
from trustml.exp1.gpy import CombinedVariance, Greedy, Variance
from trustml.exp1.tpe import TPEEvaluator

from trustml.exp1.random import Quasi, Random


def make_fig(results, is_max, save_dir: Path):
    rename_map = {
        "Quasi": "Quasi.",
        "Random": "Rand.",
        "TPEEvaluator": "TPE",
        "Greedy": "$\mathrm{GP}_\mathrm{G}$",
        "Variance": "$\mathrm{GP}_\mathrm{V}$",
        "CombinedVariance": "$\mathrm{GP}_\mathrm{UCB}$",
    }

    fig, axs = plt.subplots(1, 1, figsize=(8, 6), dpi=350)

    def make_plot_mean(df: pd.DataFrame):
        df["relative_error"] = (df["res_diff"] + df["inc_diff"]) / 2
        y = df.groupby(["n_samples"])["relative_error"].mean()
        yerr = df.groupby(["n_samples"])["relative_error"].std().values
        x = y.index.values
        y = y.values
        y_sp = make_smoothing_spline(x, y)
        y_errsp = make_smoothing_spline(x, yerr)
        x = np.linspace(x.min(), x.max(), 500)
        y = y_sp(x)
        yerr = y_errsp(x)

        sampler = rename_map[df["sampler"].iloc[0]]
        axs.fill_between(x, y, (y + yerr), alpha=0.5)
        axs.plot(x, y, label=sampler)

    def make_plot_max(df: pd.DataFrame):
        df["relative_error"] = (df["res_max_diff"] + df["inc_max_diff"]) / 2
        y = df.groupby(["n_samples"])["relative_error"].mean()
        yerr = df.groupby(["n_samples"])["relative_error"].std().values
        x = y.index.values
        y = y.values
        y_sp = make_smoothing_spline(x, y)
        y_errsp = make_smoothing_spline(x, yerr)
        x = np.linspace(x.min(), x.max(), 500)
        y = y_sp(x)
        yerr = y_errsp(x)

        sampler = rename_map[df["sampler"].iloc[0]]
        axs.fill_between(x, y, (y + yerr), alpha=0.5)
        axs.plot(x, y, label=sampler)

    axs.hlines(
        0,
        results["n_samples"].min(),
        results["n_samples"].max(),
        linestyles="dashed",
        label="Optimal",
        color="black",
    )
    results = results.set_index("sampler").loc[list(rename_map.keys())].reset_index()
    for sampler in rename_map:
        if is_max:
            make_plot_max(results[results["sampler"] == sampler])
        else:
            make_plot_mean(results[results["sampler"] == sampler])
    axs.set_ylabel("Relative Accuracy Error ($\Delta \mathcal{A}$ %)", size=20)
    axs.set_xlabel("N. Experimental Trials", size=20)
    axs.legend(fontsize=15)
    axs.tick_params(axis="both", which="major", labelsize=12)

    save_dir = save_dir.joinpath("exp1")
    save_dir.mkdir(exist_ok=True, parents=True)
    postfix = "max" if is_max else "mean"
    fig_path = save_dir.joinpath(f"figure1_{postfix}.png")
    fig.tight_layout()
    fig.savefig(fig_path)


def make_results(results: pd.DataFrame, dataset: pd.DataFrame, save_dir: Path):
    make_fig(results, is_max=True, save_dir=save_dir)
    make_fig(results, is_max=False, save_dir=save_dir)
    results["n_sample_cat"] = results["n_samples"].apply(
        lambda x: np.argmax(np.array([16, 64, np.inf]) > x)
    )
    num_cols = [
        "inc_diff",
        "res_diff",
        "inc_max_diff",
        "res_max_diff",
        "n_res",
        "n_inc",
    ]
    table_res = results.groupby(["sampler", "n_sample_cat"])[num_cols].mean()

    # Best error
    table_res["mean_err"] = (
        table_res["inc_diff"].abs() + table_res["res_diff"].abs()
    ) / 2
    # Mean error
    table_res["best_err"] = (
        table_res["inc_max_diff"].abs() + table_res["res_max_diff"].abs()
    ) / 2

    # Sampling Bias
    table_res["sampling_freq"] = table_res["n_res"] / table_res["n_inc"]
    dataset = dataset.set_index(["is_resnet"])
    resnet_bias = len(dataset.loc[True]) / len(dataset.loc[False])
    table_res["sampling_bias"] = (
        (table_res["sampling_freq"] - resnet_bias) * 100 / resnet_bias
    )

    results_table = (
        table_res.loc[
            ["Random", "TPEEvaluator", "Variance"],
            [
                "mean_err",
                "best_err",
                "sampling_bias",
            ],
        ]
        .reset_index()
        .pivot(columns="sampler", index="n_sample_cat")
    )

    pd.concat([results_table, results_table.mean().to_frame().T]).to_latex(
        save_dir.joinpath("results_table.latex"), float_format="%.2f"
    )

    table_res.loc[["CombinedVariance", "Random", "TPEEvaluator"], :]

    # DATASET TABLE

    ds_stats = pd.Series(
        dict(
            resnet_mu=dataset.loc[True]["perf"].mean(),
            renet_std=dataset.loc[True]["perf"].std(),
            inc_mu=dataset.loc[False]["perf"].mean(),
            inc_std=dataset.loc[False]["perf"].std(),
            ds_resnet_bias=len(dataset.loc[True]) / len(dataset.loc[False]),
        )
    )

    ds_stats.to_latex(save_dir.joinpath("ds_table.latex"))


def get_results():
    results_path = data_dir.joinpath("results.pickle")
    if results_path.exists():
        return pd.read_pickle(results_path)

    search_space = np.arange(1, 8)
    res = []
    n_rep = 10
    for n_points in search_space:
        for fn in [Greedy, CombinedVariance, Variance, TPEEvaluator, Quasi, Random]:
            n_trials = int(2**n_points)
            _res = fn(n_rep=n_rep, n_trials=n_trials).run()
            res.append(_res)

    res_df = pd.concat(res)
    res_df.to_pickle(results_path)
    return res_df


def run():
    results = get_results()
    dataset = get_dataset()
    make_results(results=results, dataset=dataset, save_dir=results_dir)


if __name__ == "__main__":
    run()
