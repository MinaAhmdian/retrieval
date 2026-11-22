import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from provenance import ProvenanceRecords, Unit, sha256_file
from published.frontal import population_images
from published.population import published_dir
from published.top5 import pair_metrics

EVIDENCE_COLUMNS = ["protocol", "method", "seed", "test_corpus", "archive", "rule", "query_inputs", "level", "finding",
                    "metric", "exam_id", "image_id", "value"]


class PublishedRetrieval:

    step = "published-retrieval"

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get("published.retrieval")
        self.findings = config.get("published.findings")
        self.root = published_dir(config)
        self.folder = self.root / self.settings["folder"]
        self.records = ProvenanceRecords(self.folder, self.step, {
            key: self.settings[key] for key in ("k", "protocol", "test_corpus", "archive", "rule", "query_inputs",
                                                "level", "metrics")})

    def sources(self) -> list[tuple[str, str, Path, str]]:
        out = []
        for method, spec in self.settings["methods"].items():
            folder = self.root / self.config.get(spec["folder"])
            for seed, name in spec["files"].items():
                out.append((method, str(seed), folder / f"{name}.npz", spec["key"]))
        return out

    def embeddings(self, path: Path, key: str, image_ids: pd.Series) -> np.ndarray:
        with np.load(path) as saved:
            if saved["image_ids"].tolist() != image_ids.tolist():
                raise ValueError(f"{path.name} does not hold the test images in image-identifier order")
            return saved[key]

    def rows(self, method: str, seed: str, images: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
        labels = images[self.findings].astype(int).to_numpy()
        pairs = pair_metrics(embeddings, labels, self.settings["k"])
        parts = []
        for metric, name in self.settings["metrics"].items():
            parts.append(pd.DataFrame({"finding": np.array(self.findings)[pairs["finding"]], "metric": metric,
                                       "exam_id": images["exam_id"].to_numpy()[pairs["query"]],
                                       "image_id": images["image_id"].to_numpy()[pairs["query"]],
                                       "value": pairs[name], "order": np.arange(len(pairs["query"]))}))
        table = pd.concat(parts, ignore_index=True).sort_values(["order", "metric"], kind="stable")
        fixed = {key: self.settings[key] for key in ("protocol", "test_corpus", "archive", "rule", "query_inputs",
                                                     "level")}
        return table.assign(method=method, seed=seed, **fixed)[EVIDENCE_COLUMNS]

    def unit(self, method: str, seed: str, path: Path, images: pd.DataFrame, label_rows: list[str]) -> Unit:
        return Unit(f"{method}__{seed}", ["embeddings", *images["image_id"]], [sha256_file(path), *label_rows],
                    [self.folder / f"{method}__{seed}.csv.gz"])

    def status(self) -> list[dict]:
        images = population_images(self.config)
        label_rows = images[self.findings].astype(str).agg(",".join, axis=1).tolist()
        out = []
        for method, seed, path, _ in self.sources():
            row = {"step": self.step, "unit": f"{method}__{seed}", "path": self.folder / f"{method}__{seed}.csv.gz",
                   "commit": ""}
            if not path.exists():
                out.append({**row, "status": "embeddings missing"})
                continue
            unit = self.unit(method, seed, path, images, label_rows)
            reason = self.records.stale_reason(unit)
            record = self.records.read(unit)
            if reason is None and record["code"]["uncommitted_changes"]:
                reason = "written with uncommitted changes"
            out.append({**row, "status": reason or "valid", "commit": record["code"]["commit"] if record else ""})
        return out

    def run(self) -> None:
        images = population_images(self.config)
        label_rows = images[self.findings].astype(str).agg(",".join, axis=1).tolist()
        self.folder.mkdir(parents=True, exist_ok=True)
        missing = []
        for method, seed, path, key in self.sources():
            if not path.exists():
                missing.append(f"{method} {seed}")
                continue
            unit = self.unit(method, seed, path, images, label_rows)
            if self.records.pending([unit], f"{method} {seed}"):
                table = self.rows(method, seed, images, self.embeddings(path, key, images["image_id"]))
                partial = unit.outputs[0].with_name(unit.outputs[0].name + ".partial")
                table.to_csv(partial, index=False, compression="gzip")
                os.replace(partial, unit.outputs[0])
                self.records.write(unit)
            table = pd.read_csv(unit.outputs[0], dtype={"value": float}, keep_default_na=False)
