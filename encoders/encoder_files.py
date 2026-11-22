import gzip
import json
import os

import pandas as pd

from lib.exam_encoder import LABELS
from config import Config
from encoders.retrieval import EVIDENCE_COLUMNS, PROFILE, Retrieval
from provenance import RECORD_SUFFIX, sha256_file, sha256_json, sha256_rows

RESULT_COLUMNS = ["retriever", "configuration", "backbone", "seeds", "test_corpus", "archive", "rule", "query_inputs",
                  "level", "finding", "metric", "mean", "std", "ci_low", "ci_high", "mean_raw", "n", "unit"]
GROUP = ["retriever", "test_corpus", "archive", "rule"]
UNIT_COLUMNS = ["test_corpus", "unit", "status", "commit", "rows"]


def read_evidence(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"exam_id": str}, keep_default_na=False, na_values={"value": [""]})


def result_rows(evidence: pd.DataFrame, settings: dict) -> pd.DataFrame:
    keys = [*GROUP, "level", "finding", "metric"]
    rows = evidence.groupby(keys, sort=False)["value"].agg(mean_raw="mean", n="size").reset_index()
    ap = evidence[evidence["metric"] == "ap"]
    aggregates = []
    for group, part in ap.groupby(GROUP, sort=False):
        per_finding = part.groupby("finding")["value"].mean()
        sets = {settings["macro_name"]: list(per_finding.index)}
        if set(settings["core_findings"]) <= set(per_finding.index):
            sets[settings["core_name"]] = settings["core_findings"]
        for name, findings in sets.items():
            aggregates.append({**dict(zip(GROUP, group)), "level": "finding", "finding": name,
                               "metric": settings["macro_metric"], "mean_raw": per_finding.loc[findings].mean(),
                               "n": part.loc[part["finding"].isin(findings), "exam_id"].nunique()})
    result = pd.concat([rows, pd.DataFrame(aggregates)], ignore_index=True)
    result["mean"] = result["mean_raw"] * settings["percent_scale"]
    for column, value in settings["fixed_columns"].items():
        result[column] = value
    return result[RESULT_COLUMNS]


def ordered(result: pd.DataFrame, config: Config) -> pd.DataFrame:
    settings = config.get("retrieval.files")
    evaluation = config.get("retrieval.evaluation")
    order = {
        "retriever": [*config.get("retrieval.encoders"), config.get("retrieval.classifier_retriever")],
        "test_corpus": config.get("cohort.corpora"),
        "archive": ["documented", "undocumented"],
        "rule": ["image_image", "two_view_mean", "image_report", PROFILE],
        "level": ["finding", "report"],
        "finding": [*LABELS, settings["macro_name"], settings["core_name"], "pooled"],
        "metric": ["ap", settings["macro_metric"], *[f"hit{k}" for k in evaluation["hit_ks"]],
                   *[f"ndcg{k}" for k in evaluation["ndcg_ks"]],
                   *sorted(set(result["metric"]) - {"ap", settings["macro_metric"]}
                           - {f"hit{k}" for k in evaluation["hit_ks"]} - {f"ndcg{k}" for k in evaluation["ndcg_ks"]})],
    }
    keys = pd.DataFrame({column: pd.Categorical(result[column], categories=values, ordered=True)
                         for column, values in order.items()})
    if keys.isna().any().any():
        raise ValueError("a result row has a value outside the expected categories")
    return result.loc[keys.sort_values(list(order)).index].reset_index(drop=True)


class EncoderFiles:

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get("retrieval.files")
        self.files_dir = config.path("paths.encoders_files_dir")
        self.units_csv = config.path("paths.retrieval_dir") / "evidence_units.csv"

    @property
    def evidence_path(self):
        return self.files_dir / "encoders__evidence.csv.gz"

    @property
    def result_path(self):
        return self.files_dir / "encoders__result.csv"

    def unit_status(self, job: Retrieval, spec: dict) -> tuple[str, str, object]:
        name = job.unit_name(spec)
        output = job.out_dir / f"{name}.csv.gz"
        record_path = job.out_dir / f"{name}{RECORD_SUFFIX}"
        if not record_path.exists():
            return "no record", "", output
        record = json.loads(record_path.read_text())
        mask = job.inputs.label_mask().tolist() if spec["rule"] == PROFILE else None
        ids = [f"{side}:{exam_id}" for side in ("query", "database") for exam_id in job.inputs.side(side)["exam_id"]]
        checks = [
            ("record of another step", record["step"] == job.step),
            ("queries or database differ", record["inputs"]["n"] == len(ids)
             and record["inputs"]["ids_sha256"] == sha256_rows(ids)),
            ("settings differ", record["settings_sha256"]
             == sha256_json(job.step_settings(self.config, spec["archive"], spec["rule"], mask))),
            ("output file missing", output.exists()),
            ("output file changed", output.exists() and record["outputs"].get(output.name) == sha256_file(output)),
            ("written with uncommitted changes", not record["code"]["uncommitted_changes"]),
        ]
        failed = [reason for reason, ok in checks if not ok]
        return (failed[0] if failed else "valid"), record["code"]["commit"], output

    def run(self) -> None:
        units, parts = [], []
        for corpus in self.config.get("cohort.corpora"):
            job = Retrieval(self.config, corpus)
            for spec in job.specs():
                status, commit, output = self.unit_status(job, spec)
                rows = 0
                if status == "valid":
                    part = read_evidence(output)
                    rows = len(part)
                    parts.append(part)
                units.append({"test_corpus": corpus, "unit": job.unit_name(spec), "status": status,
                              "commit": commit, "rows": rows})
        units = pd.DataFrame(units, columns=UNIT_COLUMNS)
        units.to_csv(self.units_csv, index=False)
        if (units["status"] != "valid").any():
            raise RuntimeError(f"{int((units['status'] != 'valid').sum())} evidence units are not valid; see "
                               f"{self.units_csv.name}, run the retrieval step for their corpora and run again")
        evidence = pd.concat(parts, ignore_index=True)[EVIDENCE_COLUMNS]
        self.files_dir.mkdir(parents=True, exist_ok=True)
        partial = self.evidence_path.with_name(self.evidence_path.name + ".partial")
        with open(partial, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            evidence.to_csv(handle, index=False)
        os.replace(partial, self.evidence_path)
        result = ordered(result_rows(read_evidence(self.evidence_path), self.settings), self.config)
        result.to_csv(self.result_path, index=False)
