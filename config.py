from pathlib import Path

import yaml


class Config:

    def __init__(self, path: Path):
        self.file = Path(path).resolve()
        with open(self.file) as handle:
            self.values = yaml.safe_load(handle)

    def get(self, dotted_key: str):
        value = self.values
        for part in dotted_key.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(f"{dotted_key} is not set in {self.file}")
            value = value[part]
        return value

    def path(self, dotted_key: str) -> Path:
        return Path(self.get(dotted_key))
