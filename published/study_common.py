import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd

from lib import scoring
from config import Config
from encoders.query_metrics import metric_frame
from encoders.retrieval import (DOCUMENTED, LIB_DIR, UNDOCUMENTED, Retrieval, RetrievalInputs, l2_rows,
                                        row_digest)
from provenance import ProvenanceRecords, Unit, sha256_file
from published.population import PublishedPopulation, published_dir
from published.retrieval import EVIDENCE_COLUMNS

CORPUS = "mimic"
ALIGNMENT, PROFILE, IMAGE_IMAGE = "alignment", "profile", "image_image"


class StudyCommonInputs(RetrievalInputs):

    def __init__(self, config: Config):
        super().__init__(config, CORPUS)
        population = PublishedPopulation(config)
        if population.records.stale_reason(population.unit()) is not None:
            raise RuntimeError("run published-population first: its outputs are missing or out of date")
        eligible = set(pd.read_csv(population.outputs["images.csv"], dtype=str, keep_default_na=False,
                                   usecols=["image_id"])["image_id"])
        exclusions = pd.read_csv(population.outputs["exclusions.csv"], dtype=str, keep_default_na=False)
        causes = exclusions.groupby("image_id")["cause"].agg(lambda c: ";".join(sorted(set(c))))
        selection = self.exam_images.selection()
        frontal = selection[(selection["split"] == "test") & (selection["view"] == "frontal")]
        frontal = frontal.set_index("exam_id")["image_id"]
        rows = []
        for role, side in (("query", self.queries), ("database", self.database)):
            image_ids = side["exam_id"].map(frontal)
            if image_ids.isna().any():
                raise ValueError(f"{int(image_ids.isna().sum())} {role} exams have no first frontal image")
            rows.append(pd.DataFrame({"exam_id": side["exam_id"], "role": role, "image_id": image_ids,
                                      "kept": image_ids.isin(eligible).to_numpy()}))
        table = pd.concat(rows, ignore_index=True)
        self.subset = table[table["kept"]].drop(columns="kept").reset_index(drop=True)
        self.excluded = table[~table["kept"]].drop(columns="kept").reset_index(drop=True)
        self.excluded["causes"] = self.excluded["image_id"].map(causes).fillna("not in the population")
        kept = set(self.subset["exam_id"])
        self.queries = self.queries[self.queries["exam_id"].isin(kept)].reset_index(drop=True)
        self.database = self.database[self.database["exam_id"].isin(kept)].reset_index(drop=True)
        self.first_frontal = frontal.loc[sorted(kept)]

    def frontal_rows(self, side: str, path: Path, key: str) -> np.ndarray:
        with np.load(path) as saved:
            table = pd.DataFrame(saved[key], index=saved["image_ids"])
        image_ids = self.first_frontal.loc[self.side(side)["exam_id"]]
        missing = ~image_ids.isin(table.index)
        if missing.any():
            raise ValueError(f"{path.name}: {int(missing.sum())} first frontal images have no embedding")
        return table.loc[image_ids].to_numpy()


class StudyCommon:

    step = "study-common-retrieval"

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get("published.study_common")
        self.root = published_dir(config)
        self.folder = self.root / self.settings["folder"]
        self.evidence_dir = self.folder / "evidence"
        self.inputs = StudyCommonInputs(config)
        self.retrieval = Retrieval(config, CORPUS, inputs=self.inputs)
        self.evaluation = config.get("retrieval.evaluation")
        self.archives = [DOCUMENTED, UNDOCUMENTED]
        self.report_encoder = config.get("trained_encoder.retrieval.documented_report_encoder")

    def specs(self) -> list[dict]:
        specs = []
        for method, spec in self.settings["methods"].items():
            folder = self.root / self.config.get(spec["folder"]) if "folder" in spec else None
            for seed, name in spec.get("files", {"none": None}).items():
                path = folder / f"{name}.npz" if name is not None else None
                specs += [{"method": method, "seed": str(seed), "archive": archive, "rule": spec["rule"],
                           "query_inputs": spec["query_inputs"], "key": spec.get("key"), "path": path}
                          for archive in self.archives]
        return specs

    @staticmethod
    def unit_name(spec: dict) -> str:
        return f"{spec['method']}__{spec['seed']}__{spec['archive']}__{spec['rule']}"

    def sides(self, spec: dict) -> dict:
        inputs = self.inputs
        if spec["rule"] == PROFILE:
            archive = (inputs.profile_labels() if spec["archive"] == DOCUMENTED else inputs.probabilities("database"))
            return {"query": inputs.probabilities("query"), "archive": archive, "mask": inputs.label_mask()}
        query = inputs.frontal_rows("query", spec["path"], spec["key"])
        if spec["rule"] == ALIGNMENT and spec["archive"] == DOCUMENTED:
            return {"query": query, "archive": inputs.reports(self.report_encoder)}
        archive = inputs.frontal_rows("database", spec["path"], spec["key"])
        if spec["rule"] == IMAGE_IMAGE:
            query, archive = l2_rows(query), l2_rows(archive)
        return {"query": query, "archive": archive}

    def scores(self, spec: dict, sides: dict) -> np.ndarray:
        if spec["rule"] == PROFILE:
            return scoring.profile_scores(sides["query"], sides["archive"], finding_mask=sides["mask"])
        return scoring.alignment_scores(sides["query"], sides["archive"])

    def unit(self, spec: dict, sides: dict) -> tuple[Unit, ProvenanceRecords]:
        mask = sides.get("mask")
        evaluation = {key: value for key, value in self.evaluation.items() if key != "n_jobs"}
        records = ProvenanceRecords(self.evidence_dir, self.step, {
            "evaluation": evaluation, "method": spec["method"], "seed": spec["seed"], "archive": spec["archive"],
            "rule": spec["rule"], "query_inputs": spec["query_inputs"],
            "label_mask": None if mask is None else mask.tolist(),
            "report_encoder": self.report_encoder if spec["rule"] == ALIGNMENT else None,
            "lib_sha256": {name: sha256_file(LIB_DIR / name) for name in ("metrics.py", "scoring.py")}})
        ids, contents = [], []
        for side, matrix in (("query", sides["query"]), ("database", sides["archive"])):
            entities = self.inputs.entities(side)
            for position, (exam_id, cells) in enumerate(zip(self.inputs.side(side)["exam_id"],
                                                             self.inputs.label_cells(side))):
                entity_part = ";".join(sorted("|".join(e) for e in entities[position]))
                ids.append(f"{side}:{exam_id}")
                contents.append(f"{row_digest(matrix[position])}|{cells}|{entity_part}")
        name = self.unit_name(spec)
        return Unit(name, ids, contents, [self.evidence_dir / f"{name}.csv.gz"]), records

    def evidence(self, spec: dict, scores: np.ndarray, labels: dict, report: dict) -> pd.DataFrame:
        table = metric_frame(scores, labels, report, self.evaluation)
        image_ids = "" if spec["rule"] == PROFILE else table["exam_id"].map(self.inputs.first_frontal)
        return table.assign(protocol="study_common", method=spec["method"], seed=spec["seed"], test_corpus=CORPUS,
                            archive=spec["archive"], rule=spec["rule"], query_inputs=spec["query_inputs"],
                            image_id=image_ids)[EVIDENCE_COLUMNS]

    def status(self) -> list[dict]:
        out = []
        for spec in self.specs():
            name = self.unit_name(spec)
            row = {"step": self.step, "unit": name, "path": self.evidence_dir / f"{name}.csv.gz", "commit": ""}
            if spec["path"] is not None and not spec["path"].exists():
                out.append({**row, "status": "embeddings missing"})
                continue
            unit, records = self.unit(spec, self.sides(spec))
            reason = records.stale_reason(unit)
            record = records.read(unit)
            if reason is None and record["code"]["uncommitted_changes"]:
                reason = "written with uncommitted changes"
            out.append({**row, "status": reason or "valid", "commit": record["code"]["commit"] if record else ""})
        return out

    def write_tables(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.inputs.subset.to_csv(self.folder / "subset.csv", index=False)
        self.inputs.excluded.to_csv(self.folder / "excluded.csv", index=False)

    def run(self) -> None:
        self.write_tables()
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        labels, report = self.retrieval.labels(), self.retrieval.report()
        missing = []
        for spec in self.specs():
            if spec["path"] is not None and not spec["path"].exists():
                missing.append(self.unit_name(spec))
                continue
            sides = self.sides(spec)
            unit, records = self.unit(spec, sides)
            if records.pending([unit], f"{CORPUS} {unit.name}"):
                table = self.evidence(spec, self.scores(spec, sides), labels, report)
                partial = unit.outputs[0].with_name(unit.outputs[0].name + ".partial")
                with open(partial, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
                    table.to_csv(handle, index=False)
                os.replace(partial, unit.outputs[0])
                records.write(unit)
