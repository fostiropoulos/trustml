from pathlib import Path

import gdown
import pandas as pd
from nats_bench import create
from sklearn.preprocessing import LabelEncoder

from trustml import data_dir


def get_dataset():
    """
    Method used to create the dataset used for the simulations of our experiment.
    You can use the proceprocessed dataset.pickle instead.
    """
    dataset_path = data_dir.joinpath("dataset.pickle")
    if dataset_path.exists():
        return pd.read_pickle(dataset_path)

    NATS_bench_path = Path(data_dir).joinpath("nats-bench.pickle.pbz2")

    # NATS-tss-v1_0-3ffb9.pickle.pbz2
    if not NATS_bench_path.exists():
        url = "https://drive.google.com/uc?id=1vzyK0UVH2D3fTpa1_dSWnp1gvGpAxRul"
        gdown.download(url, NATS_bench_path.as_posix(), quiet=False)

    api = create(
        NATS_bench_path.as_posix(),
        "tss",
        fast_mode=False,
        verbose=True,
    )

    data = [
        [[v]]
        + list(
            map(lambda x: list(map(lambda x: x[:-2], x.split("|")[1:-1])), k.split("+"))
        )
        for k, v in api.archstr2index.items()
    ]
    df = pd.DataFrame([[_d for d in __d for _d in d] for __d in data])

    dfs = []
    for k, arch in api.arch2infos_dict.items():
        _df = pd.DataFrame(
            [
                {**res.get_eval("ori-test"), **{"ds": k[0], "seed": k[1]}}
                for k, res in arch["200"].all_results.items()
            ]
        )
        _df["arch_id"] = k
        dfs.append(_df)
    results_df = pd.concat(dfs)
    nas_ds = results_df.set_index("arch_id").join(df)
    # https://arxiv.org/pdf/2009.00437.pdf from Fig 1. 4 corresponds to the skip connection (bottom most)
    nas_ds["is_resnet"] = nas_ds.apply(
        lambda x: "skip" in x[4] or "nor" in x[4], axis=1
    )
    nas_ds.set_index(["ds", "is_resnet"], inplace=True)
    print(
        nas_ds.loc[("cifar10", True), "accuracy"].mean()
        - nas_ds.loc[("cifar10", False), "accuracy"].mean()
    )

    x = nas_ds.loc["cifar10", 1:6].apply(LabelEncoder().fit_transform, axis=0).values
    y = nas_ds.loc["cifar10", "accuracy"].values
    dataset = pd.DataFrame(x)
    dataset["is_resnet"] = nas_ds.loc["cifar10"].index
    dataset["perf"] = y
    dataset.set_index([0, 1, 2, 3, 4, 5], inplace=True)
    dataset.to_pickle(dataset_path)
    return dataset


if __name__ == "__main__":
    dataset = get_dataset()
    raise NotImplementedError()
