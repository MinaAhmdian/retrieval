import hashlib
import json
import os
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from image_reader import ZIP_SEPARATOR

REPO_DIR = Path(__file__).resolve().parents[0]
RECORD_SUFFIX = ".provenance.json"
FIELD_SEPARATOR, ROW_SEPARATOR = "\x1f", "\x1e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_rows(*columns) -> str:
    digest = hashlib.sha256()
    for row in zip(*columns, strict=True):
        digest.update((FIELD_SEPARATOR.join(str(value) for value in row) + ROW_SEPARATOR).encode())
    return digest.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def chunked(table, size: int) -> list:
    return [(chunk, table.iloc[start:start + size]) for chunk, start in enumerate(range(0, len(table), size))]


def code_version() -> dict:
    def git(*args) -> str:
        return subprocess.run(["git", "-C", str(REPO_DIR), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no"))}


class FileDigests:

    def __init__(self, workers: int):
        self.workers = workers
        self.listings: dict[str, dict[str, str]] = {}

    def listing(self, zip_file: str) -> dict[str, str]:
        if zip_file not in self.listings:
            with zipfile.ZipFile(zip_file) as archive:
                self.listings[zip_file] = {info.filename: f"crc32:{info.CRC:08x}:{info.file_size}"
                                           for info in archive.infolist()}
        return self.listings[zip_file]

    def digest(self, path: str) -> str:
        if ZIP_SEPARATOR in path:
            zip_file, member = path.split(ZIP_SEPARATOR, 1)
            if not Path(zip_file).is_file():
                return "missing"
            return self.listing(zip_file).get(member, "missing")
        if not Path(path).is_file():
            return "missing"
        return "sha256:" + sha256_file(Path(path))

    def digests(self, paths: list[str]) -> list[str]:
        for zip_file in sorted({path.split(ZIP_SEPARATOR, 1)[0] for path in paths if ZIP_SEPARATOR in path}):
            if Path(zip_file).is_file():
                self.listing(zip_file)
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self.digest, paths))


class Unit:

    def __init__(self, name: str, ids: list, contents: list, outputs: list[Path]):
        self.name = name
        self.ids = list(ids)
        self.inputs = {"n": len(self.ids), "ids_sha256": sha256_rows(self.ids),
                       "content_sha256": sha256_rows(self.ids, contents)}
        self.outputs = list(outputs)

    def has_outputs(self) -> bool:
        return any(path.exists() for path in self.outputs)


class ProvenanceRecords:

    def __init__(self, folder: Path, step: str, settings: dict):
        self.folder = folder
        self.step = step
        self.settings = settings
        self.settings_sha256 = sha256_json(settings)
        self.code = None

    def path(self, unit: Unit) -> Path:
        return self.folder / f"{unit.name}{RECORD_SUFFIX}"

    def read(self, unit: Unit) -> dict | None:
        path = self.path(unit)
        return json.loads(path.read_text()) if path.exists() else None

    def stale_reason(self, unit: Unit) -> str | None:
        record = self.read(unit)
        if record is None:
            return "outputs without a provenance record" if unit.has_outputs() else "not computed"
        if record["step"] != self.step:
            return "record of another step"
        if record["inputs"] != unit.inputs:
            return "inputs differ"
        if record["settings_sha256"] != self.settings_sha256:
            return "settings differ"
        if sorted(record["outputs"]) != sorted(path.name for path in unit.outputs):
            return "output files differ"
        for path in unit.outputs:
            if not path.exists():
                return "output file missing"
            if sha256_file(path) != record["outputs"][path.name]:
                return "output file changed"
        return None

    def pending(self, units: list[Unit], label: str) -> list[Unit]:
        reasons = {unit.name: self.stale_reason(unit) for unit in units}
        unrecorded = [name for name, reason in reasons.items() if reason == "outputs without a provenance record"]
        if unrecorded:
            raise RuntimeError(f"[{label}] {len(unrecorded)} units in {self.folder} have outputs but no provenance "
                               "record")
        todo = [unit for unit in units if reasons[unit.name] is not None]
        for unit in todo:
            self.discard(unit)
        return todo

    def discard(self, unit: Unit) -> None:
        self.path(unit).unlink(missing_ok=True)
        for path in unit.outputs:
            path.unlink(missing_ok=True)

    def write(self, unit: Unit, run_record: dict | None = None) -> None:
        if self.code is None:
            self.code = code_version()
        record = {
            "step": self.step, "unit": unit.name, "inputs": unit.inputs,
            "settings": self.settings, "settings_sha256": self.settings_sha256,
            "outputs": {path.name: sha256_file(path) if path.exists() else "missing" for path in unit.outputs},
            "code": self.code, "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_record": run_record,
        }
        partial = self.path(unit).with_suffix(".partial")
        partial.write_text(json.dumps(record, indent=1))
        os.replace(partial, self.path(unit))
