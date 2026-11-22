import ast
from pathlib import Path

from config import Config


def source_list(path: Path, name: str) -> list[str]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise KeyError(f"{name} is not assigned at module level in {path}")


def checked_lists(config: Config) -> tuple[list[str], list[str]]:
    source = config.get("published.source")
    dataloader = Path(source["checkout"]) / source["dataloader"]
    regions, findings = config.get("published.regions"), config.get("published.findings")
    for configured, name in ((regions, source["lists"]["regions"]), (findings, source["lists"]["findings"])):
        if list(configured) != source_list(dataloader, name):
            raise ValueError(f"the configured list differs from {name} of {dataloader}")
    return list(regions), list(findings)


def environment_versions() -> dict:
    import PIL
    import torch
    import torchvision
    return {"torch": torch.__version__, "torchvision": torchvision.__version__, "pillow": PIL.__version__}
