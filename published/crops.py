import hashlib
import io
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from config import Config
from provenance import ProvenanceRecords, Unit
from published.population import PublishedPopulation, published_dir
from published.source_lists import environment_versions

RECORD_COLUMNS = ["release_path", "image_id", "exam_id", "split", "sha256", "listed_sha256", "sha256_matches",
                  "width", "height", "mode", "regions_out_of_bounds"]


def release_checksums(path: Path, expected_jpgs: int) -> dict[str, str]:
    listed, malformed = {}, 0
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and len(parts[0]) == 64:
            listed[parts[1].strip().lstrip("*")] = parts[0].lower()
        else:
            malformed += 1
    jpgs = sum(name.endswith(".jpg") for name in listed)
    if malformed or jpgs != expected_jpgs:
        raise ValueError(f"{path.name}: {malformed} malformed lines, {jpgs} JPG entries instead of {expected_jpgs}")
    return listed


def out_of_bounds(boxes: np.ndarray, width: int, height: int) -> int:
    x, y, w, h = boxes.T
    return int(((x < 0) | (y < 0) | (x + w > width) | (y + h > height)).sum())


def png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


_RESIZE = None


def _setup(size: list[int], antialias: bool) -> None:
    global _RESIZE
    from torchvision import transforms
    _RESIZE = transforms.Resize(tuple(size), antialias=antialias)


def crop_original(data: bytes, boxes: np.ndarray, resize) -> tuple[dict, list[bytes]]:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        crops = [png(resize(image.crop((x, y, x + w, y + h)))) for x, y, w, h in boxes.tolist()]
        crops.append(png(resize(image)))
        return {"width": width, "height": height, "mode": image.mode,
                "regions_out_of_bounds": out_of_bounds(boxes, width, height)}, crops


def _crop_task(task: tuple[list[str], list[str], np.ndarray]) -> list[tuple[dict, list[bytes]]]:
    files, expected, boxes = task
    results = []
    for file, listed, image_boxes in zip(files, expected, boxes):
        data = Path(file).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != listed:
            results.append(({"sha256": sha}, []))
            continue
        fields, crops = crop_original(data, image_boxes, _RESIZE)
        results.append(({"sha256": sha, **fields}, crops))
    return results


class PublishedCrops:

    step = "published-crops"

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get("published.crops")
        self.originals = config.get("published.originals")
        self.folder = published_dir(config) / self.settings["folder"]
        self.population = PublishedPopulation(config)
        self.records = ProvenanceRecords(self.folder, self.step, {
            "crops": {k: self.settings[k] for k in ("resize", "antialias", "members")},
            "regions": self.population.regions, "release": self.originals["release"]})

    def outputs(self, chunk: int) -> list[Path]:
        return [self.folder / f"crops_{chunk:03d}.zip", self.folder / f"originals_{chunk:03d}.csv"]

    def units(self, images: pd.DataFrame, boxes: np.ndarray) -> list[tuple[Unit, pd.DataFrame, np.ndarray]]:
        units = []
        for chunk, part in images.groupby("chunk", sort=True):
            rows = part.index.to_numpy()
            contents = [f"{path}|{box.tolist()}" for path, box in zip(part["release_path"], boxes[rows])]
            units.append((Unit(f"chunk_{chunk:03d}", part["image_id"], contents, self.outputs(int(chunk))),
                          part, boxes[rows]))
        return units

    def run(self) -> None:
        self.versions = environment_versions()
        population = self.population.outputs
        if self.population.records.stale_reason(self.population.unit()) is not None:
            raise RuntimeError("run published-population first: its outputs are missing or out of date")
        images = pd.read_csv(population["images.csv"], dtype=str, keep_default_na=False)
        images["chunk"] = images["chunk"].astype(int)
        with np.load(population["regions.npz"]) as saved:
            if saved["image_id"].tolist() != images["image_id"].tolist():
                raise ValueError("regions.npz and images.csv list different images")
            boxes = saved["boxes"]
        self.folder.mkdir(parents=True, exist_ok=True)
        root = Path(self.originals["root"])
        units = self.units(images, boxes)
        todo = {unit.name for unit in self.records.pending([unit for unit, _, _ in units], "mimic")}
        listed = None
        waiting, left, failed, written, limit = 0, 0, [], 0, self.settings["chunks_per_job"]
        for unit, part, part_boxes in units:
            if unit.name not in todo:
                continue
            if limit is not None and written + len(failed) >= limit:
                left += 1
                continue
            files = [root / path for path in part["release_path"]]
            absent = sum(not file.is_file() for file in files)
            if absent:
                waiting += 1
                continue
            if listed is None:
                listed = release_checksums(root / self.originals["checksums"], self.originals["expected_jpgs"])
            mismatched = self.write_chunk(unit, part, part_boxes, files, listed)
            if mismatched:
                failed.append(unit.name)
            else:
                written += 1
        if failed:
            raise RuntimeError(f"{len(failed)} chunks have originals whose SHA-256 differs from the release list; "
                               "download them again")

    def write_chunk(self, unit: Unit, part: pd.DataFrame, boxes: np.ndarray, files: list[Path],
                    listed: dict[str, str]) -> int:
        expected = [listed.get(path, "") for path in part["release_path"]]
        size = self.settings["images_per_task"]
        tasks = [([str(f) for f in files[i:i + size]], expected[i:i + size], boxes[i:i + size])
                 for i in range(0, len(files), size)]
        zip_path, table_path = unit.outputs
        partial = zip_path.with_suffix(".partial")
        members = self.settings["members"]
        rows, mismatched = [], 0
        with ProcessPoolExecutor(self.settings["workers"], initializer=_setup,
                                 initargs=(self.settings["resize"], self.settings["antialias"])) as pool, \
                zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
            results = (result for part_results in pool.map(_crop_task, tasks) for result in part_results)
            for (_, image), (fields, crops), listed_sha in zip(part.iterrows(), results, expected):
                matches = fields["sha256"] == listed_sha
                if not matches:
                    mismatched += 1
                    continue
                for r, data in enumerate(crops[:-1]):
                    archive.writestr(members["region"].format(image_id=image["image_id"], region=r), data)
                archive.writestr(members["image"].format(image_id=image["image_id"]), crops[-1])
                rows.append({"release_path": image["release_path"], "image_id": image["image_id"],
                             "exam_id": image["exam_id"], "split": image["split"], "listed_sha256": listed_sha,
                             "sha256_matches": matches, **fields})
        if mismatched:
            partial.unlink(missing_ok=True)
            return mismatched
        expected_members = len(rows) * (len(self.population.regions) + 1)
        with zipfile.ZipFile(partial) as archive:
            if len(archive.namelist()) != expected_members or archive.testzip() is not None:
                raise RuntimeError(f"[{unit.name}] the written zip does not read back complete")
        partial.replace(zip_path)
        pd.DataFrame(rows)[RECORD_COLUMNS].to_csv(table_path, index=False)
        self.records.write(unit, run_record={"versions": self.versions})
        return 0
