import hashlib
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd

from config import Config
from provenance import ProvenanceRecords, Unit, sha256_file

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
HF_URL = "https://huggingface.co"


class TaixExport:

    step = "export-taix"

    def __init__(self, config: Config):
        settings = config.get("corpora.taix")
        self.repo, self.revision = settings["hf_repo"], settings["hf_revision"]
        self.keep_parquet = settings["keep_parquet"]
        self.shards_dir = config.path("corpora.taix.parquet_dir")
        self.images_dir = config.path("corpora.taix.images_dir")
        self.annotation = config.path("corpora.taix.annotation")
        self.export_dir = config.path("corpora.taix.export_dir")
        self.batch_size = settings["export_batch_size"]
        self.records = ProvenanceRecords(self.export_dir, self.step,
                                         {"repo": self.repo, "revision": self.revision,
                                          "image_bytes": "as stored in the parquet"})

    def released_shards(self) -> list[dict]:
        url = f"{HF_URL}/api/datasets/{self.repo}/tree/{self.revision}/data"
        with urllib.request.urlopen(url, timeout=60) as response:
            listing = json.load(response)
        shards = [{"path": item["path"], "size": item["lfs"]["size"], "sha256": item["lfs"]["oid"]}
                  for item in listing if item["path"].endswith(".parquet")]
        return sorted(shards, key=lambda shard: shard["path"])

    def unit(self, shard: dict) -> Unit:
        name = Path(shard["path"]).name
        return Unit(Path(name).stem, [name], [shard["sha256"]], [self.export_dir / f"{Path(name).stem}.csv"])

    def images_unchanged(self, unit: Unit) -> bool:
        rows = pd.read_csv(unit.outputs[0], dtype=str, keep_default_na=False, usecols=["UID", "png_sha256"])
        return all((self.images_dir / f"{uid}.png").is_file() and sha256_file(self.images_dir / f"{uid}.png") == digest
                   for uid, digest in zip(rows["UID"], rows["png_sha256"]))

    def download(self, shard: dict) -> Path:
        target = self.shards_dir / Path(shard["path"]).name
        if target.is_file() and sha256_file(target) == shard["sha256"]:
            return target
        partial = target.with_suffix(".partial")
        url = f"{HF_URL}/datasets/{self.repo}/resolve/{self.revision}/{shard['path']}"
        with urllib.request.urlopen(url, timeout=300) as response, open(partial, "wb") as handle:
            for block in iter(lambda: response.read(1 << 24), b""):
                handle.write(block)
        if sha256_file(partial) != shard["sha256"]:
            partial.unlink()
            raise ValueError(f"{target.name}: downloaded file differs from its released SHA-256")
        os.replace(partial, target)
        return target

    def run(self) -> None:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        shards = self.released_shards()
        seen: set[str] = set()
        tables = []
        for index, shard in enumerate(shards, start=1):
            unit = self.unit(shard)
            if self.records.stale_reason(unit) is None and self.images_unchanged(unit):
                table = pd.read_csv(unit.outputs[0], dtype=str, keep_default_na=False)
            else:
                self.records.discard(unit)
                path = self.download(shard)
                table, written = self.export_shard(path, unit)
                self.records.write(unit)
                if not self.keep_parquet:
                    path.unlink()
            duplicated = seen & set(table["UID"])
            if duplicated:
                raise ValueError(f"{len(duplicated)} UIDs of {unit.ids[0]} are in an earlier shard")
            seen |= set(table["UID"])
            tables.append(table.drop(columns="png_sha256"))
        annotation = pd.concat(tables, ignore_index=True)
        annotation.to_csv(self.annotation, index=False)

    def export_shard(self, shard: Path, unit: Unit) -> tuple[pd.DataFrame, int]:
        import pyarrow.parquet as pq

        rows, written = [], 0
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=self.batch_size):
            for record in batch.to_pylist():
                data = record.pop("Image")["bytes"]
                if not data.startswith(PNG_MAGIC):
                    raise ValueError(f"{shard.name}: UID {record['UID']} is not PNG")
                target = self.images_dir / f"{record['UID']}.png"
                if not (target.is_file() and target.read_bytes() == data):
                    target.write_bytes(data)
                    written += 1
                rows.append({**record, "png_sha256": hashlib.sha256(data).hexdigest()})
        table = pd.DataFrame(rows)
        if table["UID"].duplicated().any():
            raise ValueError(f"duplicate UID in {shard.name}")
        table.to_csv(unit.outputs[0], index=False)
        return pd.read_csv(unit.outputs[0], dtype=str, keep_default_na=False), written
