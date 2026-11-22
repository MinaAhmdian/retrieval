from pathlib import Path

from config import Config

RUN_GROUPS = {"trained_encoder.runs": "paths.trained_dir", "controls.runs": "paths.controls_dir",
              "transfer.runs": "paths.transfer_dir"}
DEFAULT_GROUP = "trained_encoder.runs"


def run_names(config: Config, group: str = DEFAULT_GROUP) -> list[str]:
    runs = config.get(group)
    return [f"{backbone}__{configuration}__{seed}" for backbone, configurations in runs["configurations"].items()
            for configuration in configurations for seed in runs["seeds"]]


def all_run_names(config: Config) -> list[str]:
    names = [name for group in RUN_GROUPS for name in run_names(config, group)]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"runs in more than one group: {repeated}")
    return names


def run_root(config: Config, run: str) -> Path:
    for group, root in RUN_GROUPS.items():
        if run in run_names(config, group):
            return config.path(root)
    raise ValueError(f"{run} is not a run of {list(RUN_GROUPS)}")
