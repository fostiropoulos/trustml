import argparse
import typing as ty
from pathlib import Path

import numpy as np
import ray
import torchvision
from ablator import (
    Derived,
    ModelConfig,
    ModelWrapper,
    Optional,
    ParallelConfig,
    ParallelTrainer,
    TrainConfig,
    configclass,
)
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

import trustml as trustml_module
from trustml import config_path, package_dir


def acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


class CatDog(torchvision.datasets.CIFAR10):
    """
    Sticking with the Cartoon theme.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: ty.Optional[ty.Callable] = None,
        target_transform: ty.Optional[ty.Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        cat_label = self.class_to_idx["cat"]
        dog_label = self.class_to_idx["dog"]
        np_targets = np.array(self.targets)
        mask = (np_targets == cat_label) | (np_targets == dog_label)
        self.class_to_idx = {"cat": 0, "dog": 1}

        self.targets = np_targets[mask]
        self.targets[self.targets == cat_label] = 0
        self.targets[self.targets == dog_label] = 1
        self.data = self.data[mask]

    def __getitem__(self, index: int) -> dict[ty.Any, ty.Any]:
        x, y = super().__getitem__(index)
        return {
            "x": x,
            "y_true": y,
        }


@configclass
class CDModelConfig(ModelConfig):
    num_classes: Derived[Optional[int]] = None


@configclass
class CDRun(ParallelConfig):
    model_config: CDModelConfig
    train_config: TrainConfig


def cat_dog(config: CDRun, flag="train") -> DataLoader:
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.AugMix(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    root = config.train_config.dataset
    dataset = CatDog(
        root=root,
        train=flag == "train",
        transform=data_transforms[flag],
        download=not Path(root).exists(),
    )
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=config.train_config.batch_size,
        shuffle=True,
    )
    return dataloader


class CDModel(nn.Module):
    def __init__(self, config: CDModelConfig) -> None:
        super().__init__()

        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=config.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y_true=None):
        loss = None

        outputs = self.model(x)
        loss = self.loss_fn(outputs, y_true)

        out = outputs.argmax(dim=-1)
        return {"y_pred": out, "y_true": y_true}, loss


class CDWrapper(ModelWrapper):
    def __init__(self):
        super().__init__(CDModel)

    def make_dataloader_train(self, run_config: CDRun):  # type: ignore
        return cat_dog(run_config, flag="train")

    def make_dataloader_val(self, run_config: CDRun):  # type: ignore
        return cat_dog(run_config, flag="val")

    def config_parser(self, run_config: CDRun):  # type: ignore
        run_config.model_config.num_classes = 2
        return run_config

    def evaluation_functions(self) -> dict[str, ty.Callable]:
        return {"accuracy_score": acc}


def train(save_dir: str | None = None):
    if save_dir is None:
        save_dir = package_dir.joinpath("save-dir")
    run_config = CDRun.load(config_path)
    run_config.experiment_dir = save_dir
    model = CDWrapper()

    # NOTE: MultiProcess trainer

    trainer = ParallelTrainer(
        wrapper=model,
        run_config=run_config,
    )
    ray.init()
    trainer.launch(
        working_directory=package_dir, auxilary_modules=[trustml_module], resume=True
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--save_dir", type=str, required=False)
    kwargs = vars(args.parse_args())
    config = train(**kwargs)
