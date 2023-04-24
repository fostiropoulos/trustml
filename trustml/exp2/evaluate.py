import pandas as pd
import scipy as sp
import seaborn as sns
import numpy as np


from trustml import data_dir, results_dir

save_dir = results_dir.joinpath("exp2")
save_dir.mkdir(exist_ok=True, parents=True)
results_tpe = pd.read_csv(data_dir.joinpath(f"results_tpe.csv"), index_col=False)
results_tpe = results_tpe.groupby("path").apply(
    lambda x: x.sort_values("val_accuracy_score", na_position="first").iloc[-1]
)
results_random = pd.read_csv(data_dir.joinpath(f"results_random.csv"), index_col=False)
results_random = results_random.groupby("path").apply(
    lambda x: x.sort_values("val_accuracy_score", na_position="first").iloc[-1]
)


config_name_remap = {
    "train_config.optimizer_config.arguments.lr": "Learning Rate",
    "train_config.batch_size": "Batch Size",
}
metric_name_map = {"val_loss": "Val. Loss.", "val_accuracy_score": "Val. Accuracy"}


def make_figures():

    sns.set(font_scale=1)  # crazy big

    sns.set_style("whitegrid")

    for metric in metric_name_map:
        for config in config_name_remap:

            dfs = []
            for name, res in zip(["TPE", "Random"], [results_tpe, results_random]):
                _df = res[[metric, config]].reset_index(drop=True)
                _df.columns = [metric_name_map[metric], config_name_remap[config]]
                r, p = sp.stats.pearsonr(_df.iloc[:, 0], _df.iloc[:, 1])
                _df["Sampling Method"] = f"{name}\n$\\rho$={r:.2f}"
                dfs.append(_df)

            df = pd.concat(dfs)

            g = sns.lmplot(
                df,
                x=df.columns[1],
                y=df.columns[0],
                markers=".",
                hue=df.columns[2],
                scatter_kws={"alpha": 0.6},
            )
            means = df.groupby(df.columns[2]).mean().iloc[:, 1]
            max_val = df.max()[0]
            min_val = df.min()[0]
            for line in g.axes[0][0].lines:
                line.set_label(s="")
            tpe_line = g.axes[0][0].lines[0]._color
            rand_line = g.axes[0][0].lines[1]._color
            if config == "train_config.batch_size":
                label_mean_1 = f"{means[1]:.2f}"
                label_mean_0 = f"{means[0]:.2f}"
            else:
                label_mean_1 = f"{means[1]:.4f}"
                label_mean_0 = f"{means[0]:.4f}"

            h1 = g.axes[0][0].vlines(
                means[1],
                min_val,
                max_val,
                linestyle="dashed",
                color=tpe_line,
                label=label_mean_1,
            )
            h0 = g.axes[0][0].vlines(
                means[0],
                min_val,
                max_val,
                linestyle="dashed",
                color=rand_line,
                label=label_mean_0,
            )
            g.axes[0][0].legend(handles=[h1, h0], title="Mean")
            g.figure.savefig(save_dir.joinpath(f"{metric}_{config}.png"))


if __name__ == "__main__":
    make_figures()
    best_perf_rand = results_random.max()["val_accuracy_score"]
    best_perf_tpe = results_tpe.max()["val_accuracy_score"]
    best_perf = pd.Series({"tpe": best_perf_tpe, "rand": best_perf_rand})
    best_perfs = []
    for i in range(10):
        trials_rand = np.random.permutation(len(results_random))[:16]
        trials_tpe = np.random.permutation(len(results_tpe))[:16]
        sampled_perf = pd.Series(
            {
                "tpe": results_tpe.iloc[trials_tpe].max()["val_accuracy_score"],
                "rand": results_random.iloc[trials_rand].max()["val_accuracy_score"],
            }
        )
        best_perfs.append(sampled_perf)
    df = pd.DataFrame(best_perfs)
    print(f"{best_perf} {best_perf_tpe} {df.max()}")
    results_tpe[list(config_name_remap.keys()) + list(metric_name_map.keys())]
    results_random[list(config_name_remap.keys())].mean()
    pass
