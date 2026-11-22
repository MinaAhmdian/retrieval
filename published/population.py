import json
import math
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image

from config import Config
from provenance import FileDigests, ProvenanceRecords, Unit
from published.source_lists import checked_lists

BOX_FIELDS = ["original_x1", "original_y1", "original_width", "original_height"]
GRAPH_SUFFIX = "_SceneGraph.json"
IMAGE_COLUMNS = ["image_id", "exam_id", "patient_id", "split", "view", "release_path", "chunk"]
EXCLUSION_COLUMNS = ["image_id", "exam_id", "patient_id", "split", "view", "cause", "regions"]


def drop_unavailable(eligible: pd.DataFrame, boxes: np.ndarray, labels: np.ndarray, paths: set[str], cause: str):
    unknown = paths - set(eligible["release_path"])
    if unknown:
        raise ValueError(f"{len(unknown)} unavailable originals are not eligible images")
    drop = eligible["release_path"].isin(paths).to_numpy()
    rows = eligible.loc[drop, EXCLUSION_COLUMNS[:5]].assign(cause=cause, regions="")
    return eligible[~drop].reset_index(drop=True), boxes[~drop], labels[~drop], rows


def published_dir(config: Config):
    return config.path("paths.published_dir")


def release_path(patient_id: str, exam_id: str, image_id: str) -> str:
    return f"files/p{patient_id[:2]}/p{patient_id}/s{exam_id}/{image_id}.jpg"


def crop_size(box) -> tuple[int, int] | None:
    x, y, w, h = box
    try:
        return Image.new("L", (1, 1)).crop((x, y, x + w, y + h)).size
    except ValueError:
        return None


def box_cause(box) -> str | None:
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        return "nonfinite box"
    if box[2] <= 0 or box[3] <= 0:
        return "nonpositive width or height"
    size = crop_size(box)
    if size is None or min(size) <= 0:
        return "empty crop"
    return None


def object_region(obj: dict) -> str:
    return str(obj.get("object_id", "")).split("_")[1] if "_" in str(obj.get("object_id", "")) else ""


def read_graph(graph: dict, regions: list[str], findings: list[str]) -> dict:
    by_region: dict[str, list[dict]] = {}
    renamed = 0
    for obj in graph.get("objects", []):
        region = object_region(obj)
        by_region.setdefault(region, []).append(obj)
        renamed += region in regions and obj.get("bbox_name") != region
    boxes = np.zeros((len(regions), 4))
    labels = np.zeros((len(regions), len(findings)), dtype=np.uint8)
    index = {finding: i for i, finding in enumerate(findings)}
    causes, duplicates = [], 0
    for r, region in enumerate(regions):
        objects = by_region.get(region, [])
        if not objects:
            causes.append(("missing region", region))
            continue
        if len(objects) > 1:
            if len({json.dumps([o.get(k) for k in ["object_id", *BOX_FIELDS]]) for o in objects}) > 1:
                causes.append(("ambiguous duplicate region", region))
                continue
            duplicates += 1
        box = [objects[0].get(k) for k in BOX_FIELDS]
        cause = box_cause(box)
        if cause:
            causes.append((cause, region))
            continue
        boxes[r] = box
        object_id = objects[0].get("object_id")
        for attribute in graph.get("attributes", []):
            if attribute.get("object_id") != object_id:
                continue
            for mentions in attribute.get("attributes", []):
                for mention in mentions:
                    parts = str(mention).split("|")
                    if len(parts) > 2 and parts[2] in index:
                        labels[r, index[parts[2]]] = 1 if parts[1] == "yes" else 0
    return {"boxes": boxes, "labels": labels, "causes": causes, "identical_duplicates": duplicates,
            "bbox_name_differs": int(renamed), "graph_image_id": str(graph.get("image_id", "")),
            "graph_exam_id": str(graph.get("study_id", "")), "graph_patient_id": str(graph.get("patient_id", "")),
            "attribute_records": len(graph.get("attributes", []))}


_ARCHIVE = None


def _open(path: str) -> None:
    global _ARCHIVE
    _ARCHIVE = zipfile.ZipFile(path)


def _read(task: tuple[list[str], list[str], list[str]]) -> list[dict]:
    members, regions, findings = task
    return [read_graph(json.loads(_ARCHIVE.read(member)), regions, findings) for member in members]


class PublishedPopulation:

    step = "published-population"

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.get("published.population")
        self.folder = published_dir(config) / self.settings["folder"]
        self.regions, self.findings = checked_lists(config)
        self.scene_graphs = config.get("published.scene_graphs")
        self.records = ProvenanceRecords(self.folder, self.step, {
            "population": self.settings, "regions": self.regions, "findings": self.findings,
            "views": config.get("published.views"), "split_map": config.get("corpora.mimic.split_map"),
            "chunks": config.get("published.crops.chunks"),
            "chunk_size": config.get("published.crops.chunk_size"),
            **({"unavailable_cause": config.get("published.originals.unavailable.cause")}
               if config.get("published.originals.unavailable") is not None else {})})

    @property
    def outputs(self) -> dict:
        names = ["images.csv", "regions.npz", "exclusions.csv", "counts.csv"]
        return {name: self.folder / name for name in names}

    def unit(self) -> Unit:
        files = [self.config.get("corpora.mimic.metadata"), self.config.get("corpora.mimic.split"), self.scene_graphs,
                 self.config.get("published.images_to_avoid")]
        unavailable = self.config.get("published.originals.unavailable")
        if unavailable is not None:
            files.append(unavailable["list"])
        return Unit("population", files, FileDigests(len(files)).digests(files), list(self.outputs.values()))

    def run(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        unit = self.unit()
        if not self.records.pending([unit], "mimic"):
            return
        images, regions, exclusions, counts = self.build()
        images.to_csv(self.outputs["images.csv"], index=False)
        np.savez(self.outputs["regions.npz"], image_id=images["image_id"].to_numpy(dtype=str), **regions)
        exclusions.to_csv(self.outputs["exclusions.csv"], index=False)
        counts.to_csv(self.outputs["counts.csv"], index=False)
        self.records.write(unit)

    def metadata_images(self) -> pd.DataFrame:
        from corpora.mimic import MimicCorpus
        images = MimicCorpus(self.config).images()
        images = images.rename(columns={"view_label": "view"})[["image_id", "exam_id", "patient_id", "split", "view"]]
        images["release_path"] = [release_path(*row) for row in
                                  images[["patient_id", "exam_id", "image_id"]].itertuples(index=False)]
        return images.reset_index(drop=True)

    def graph_members(self) -> dict[str, str]:
        with zipfile.ZipFile(self.scene_graphs) as archive:
            return {PurePosixPath(name).name[:-len(GRAPH_SUFFIX)]: name for name in archive.namelist()
                    if name.endswith(GRAPH_SUFFIX)}

    def read_graphs(self, members: list[str]) -> list[dict]:
        size = self.settings["graphs_per_task"]
        tasks = [(members[i:i + size], self.regions, self.findings) for i in range(0, len(members), size)]
        with ProcessPoolExecutor(self.settings["workers"], initializer=_open, initargs=(self.scene_graphs,)) as pool:
            return [graph for part in pool.map(_read, tasks) for graph in part]

    def build(self) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
        images = self.metadata_images()
        members = self.graph_members()
        views = set(self.config.get("published.views"))
        in_view = images["view"].isin(views)
        has_graph = images["image_id"].isin(members.keys())
        candidates = images[in_view & has_graph].reset_index(drop=True)
        graphs = self.read_graphs([members[i] for i in candidates["image_id"]])
        causes = [list(graph["causes"]) for graph in graphs]
        for c, graph in enumerate(graphs):
            row = candidates.iloc[c]
            if graph["graph_image_id"] != row["image_id"]:
                causes[c].insert(0, ("scene graph image_id differs", ""))
        whole = [images.loc[mask, EXCLUSION_COLUMNS[:5]].assign(cause=cause, regions="")
                 for mask, cause in ((~in_view, "view not PA or AP"), (in_view & ~has_graph, "no scene graph"))]
        exclusion_rows = []
        for c, image_causes in enumerate(causes):
            grouped: dict[str, list[str]] = {}
            for cause, region in image_causes:
                grouped.setdefault(cause, []).append(region)
            for cause, names in grouped.items():
                exclusion_rows.append({**candidates.loc[c, EXCLUSION_COLUMNS[:5]].to_dict(), "cause": cause,
                                       "regions": ";".join(name for name in names if name)})
        exclusions = pd.concat([*whole, pd.DataFrame(exclusion_rows, columns=EXCLUSION_COLUMNS)],
                               ignore_index=True)[EXCLUSION_COLUMNS]
        keep = [c for c, image_causes in enumerate(causes) if not image_causes]
        eligible = candidates.iloc[keep].copy()
        boxes = np.stack([graphs[c]["boxes"] for c in keep]) if keep else np.zeros((0, len(self.regions), 4))
        labels = (np.stack([graphs[c]["labels"] for c in keep]) if keep
                  else np.zeros((0, len(self.regions), len(self.findings)), dtype=np.uint8))
        eligible["empty_attributes"] = [graphs[c]["attribute_records"] == 0 for c in keep]
        for f, finding in enumerate(self.findings):
            eligible[finding] = labels[:, :, f].max(axis=1).astype(int) if keep else []
        diagnostics = self.diagnostics(eligible)
        order, eligible = self.chunk_order(eligible)
        boxes, labels = boxes[order], labels[order]
        unavailable = self.config.get("published.originals.unavailable")
        if unavailable is not None:
            paths = set(Path(unavailable["list"]).read_text().split())
            eligible, boxes, labels, dropped = drop_unavailable(eligible, boxes, labels, paths, unavailable["cause"])
            exclusions = pd.concat([exclusions, dropped], ignore_index=True)[EXCLUSION_COLUMNS]
        stats = {"identical_duplicates_kept": sum(g["identical_duplicates"] for g in graphs),
                 "objects_whose_bbox_name_differs": sum(g["bbox_name_differs"] for g in graphs),
                 "graphs_whose_study_id_differs": sum(g["graph_exam_id"] != candidates.loc[c, "exam_id"]
                                                      for c, g in enumerate(graphs)),
                 "graphs_whose_patient_id_differs": sum(g["graph_patient_id"] != candidates.loc[c, "patient_id"]
                                                        for c, g in enumerate(graphs)),
                 "graphs_without_a_metadata_image": len(set(members) - set(images["image_id"])),
                 "graphs_of_other_views": int((~in_view & has_graph).sum())}
        counts = pd.concat([self.counts(images, in_view, has_graph, eligible, exclusions, stats), diagnostics],
                           ignore_index=True)
        return eligible, {"boxes": boxes, "labels": labels}, exclusions, counts

    def diagnostics(self, eligible: pd.DataFrame) -> pd.DataFrame:
        avoid = set(pd.read_csv(self.config.get("published.images_to_avoid"), dtype=str,
                                usecols=["dicom_id"])["dicom_id"])
        flags = pd.DataFrame({"split": eligible["split"], "empty": eligible["empty_attributes"],
                              "avoid": eligible["image_id"].isin(avoid)})
        rows = []
        for split in [*self.config.get("published.reference_counts"), "all"]:
            part = flags if split == "all" else flags[flags["split"] == split]
            for quantity, value in (("eligible_without_attribute_records", part["empty"].sum()),
                                    ("eligible_in_images_to_avoid", part["avoid"].sum()),
                                    ("eligible_in_neither", (~part["empty"] & ~part["avoid"]).sum())):
                rows.append({"split": split, "quantity": f"diagnostic: {quantity}", "value": int(value)})
        return pd.DataFrame(rows)

    def chunk_order(self, eligible: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        settings = self.config.get("published.crops")
        positions, chunks, number = [], [], 0
        eligible = eligible.reset_index(drop=True)
        for group in settings["chunks"]:
            part = eligible[eligible["split"].isin(group)].sort_values("release_path", kind="stable")
            for start in range(0, len(part), settings["chunk_size"]):
                rows = part.index[start:start + settings["chunk_size"]]
                positions += list(rows)
                chunks += [number] * len(rows)
                number += 1
        if len(positions) != len(eligible):
            raise ValueError("published.crops.chunks does not cover every split of the eligible images")
        ordered = eligible.loc[positions].reset_index(drop=True)
        ordered["chunk"] = chunks
        return np.array(positions, dtype=int), ordered[IMAGE_COLUMNS + self.findings]

    def counts(self, images, in_view, has_graph, eligible, exclusions, stats) -> pd.DataFrame:
        reference = self.config.get("published.reference_counts")
        rows = []
        for split in [*reference, "all"]:
            chosen = images["split"].eq(split) if split != "all" else images["split"].notna()
            part = eligible["split"].eq(split) if split != "all" else eligible["split"].notna()
            excluded = exclusions[exclusions["split"].eq(split)] if split != "all" else exclusions
            values = {"metadata_images": int(chosen.sum()), "pa_or_ap": int((chosen & in_view).sum()),
                      "pa_or_ap_with_scene_graph": int((chosen & in_view & has_graph).sum()),
                      "eligible": int(part.sum())}
            for cause, group in excluded.groupby("cause"):
                values[f"images_excluded_for: {cause}"] = int(group["image_id"].nunique())
            if split != "all":
                values["published_reference"] = int(reference[split])
                values["eligible_minus_reference"] = int(part.sum()) - int(reference[split])
            rows += [{"split": split, "quantity": q, "value": v} for q, v in values.items()]
        rows += [{"split": "all", "quantity": q, "value": int(v)} for q, v in stats.items()]
        return pd.DataFrame(rows)
