from pathlib import Path

__version__ = "0.0.1"

package_dir = Path().cwd()
data_dir = package_dir.joinpath("data")
config_path = package_dir.joinpath("config.yaml")
results_dir = package_dir.joinpath("results")


assert (
    data_dir.exists() and config_path.exists() and results_dir.exists()
), "Must run module inside the git-repo."
