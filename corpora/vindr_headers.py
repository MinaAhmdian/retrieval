import io
import netrc
import time
import warnings
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import pandas as pd
import pydicom
import requests

from config import Config
from provenance import ProvenanceRecords, Unit, chunked

PIXEL_DATA_TAG = b"\xe0\x7f\x10\x00"
HEADER_FIELDS = ["PatientAge", "PatientSex", "ViewPosition"]
warnings.filterwarnings("ignore", message="Expected implicit VR")


class VinDrHeaders:

    step = "fetch-vindr-headers"

    def __init__(self, config: Config):
        self.settings = config.get("corpora.vindr.release_headers")
        self.out_dir = config.path("paths.cohort_dir") / "vindr" / "release_headers"
        self.session = None
        self.records = ProvenanceRecords(self.out_dir, self.step, self.step_settings(config, "vindr"))

    @staticmethod
    def step_settings(config: Config, corpus_name: str) -> dict:
        settings = config.get("corpora.vindr.release_headers")
        return {key: settings[key] for key in ("url", "header_bytes", "chunk_size")}

    def dicom_files(self) -> pd.DataFrame:
        with open(self.settings["file_list"]) as handle:
            rows = [line.split()[:2] for line in handle if line.strip()]
        files = pd.DataFrame(rows, columns=["sha256", "path"])
        return files[files["path"].str.endswith(".dicom")].sort_values("path").reset_index(drop=True)

    def units(self) -> list[Unit]:
        return [Unit(f"headers_{chunk:05d}", part["path"], part["sha256"], [self.out_dir / f"headers_{chunk:05d}.csv"])
                for chunk, part in chunked(self.dicom_files(), self.settings["chunk_size"])]


    def connect(self) -> None:
        login, _, password = netrc.netrc().authenticators(urlparse(self.settings["url"]).hostname)
        self.session = requests.Session()
        self.session.auth = (login, password)
        self.session.headers["User-Agent"] = self.settings["user_agent"]

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        units = self.units()
        todo = self.records.pending(units, "vindr")
        if todo:
            self.connect()
        for unit in todo:
            pd.DataFrame([self.header(path) for path in unit.ids]).to_csv(unit.outputs[0], index=False)
            self.records.write(unit)
        self.table()

    def header(self, path: str) -> dict:
        response = self.fetch(path)
        if response.status_code != 206:
            raise RuntimeError(f"HTTP {response.status_code} for a release DICOM")
        if PIXEL_DATA_TAG not in response.content:
            raise RuntimeError(f"the header of a release DICOM is longer than {self.settings['header_bytes']} bytes")
        dataset = pydicom.dcmread(io.BytesIO(response.content), stop_before_pixels=True, force=True)
        time.sleep(self.settings["pause_seconds"])
        return {"image_id": Path(path).stem, "split": PurePosixPath(path).parts[0],
                **{field: str(dataset.get(field, "")).strip() for field in HEADER_FIELDS}}

    def fetch(self, path: str) -> requests.Response:
        return self.get(path, {"Range": f"bytes=0-{self.settings['header_bytes'] - 1}"})

    def fetch_whole(self, path: str) -> requests.Response:
        return self.get(path, {})

    def get(self, path: str, headers: dict) -> requests.Response:
        for attempt in range(self.settings["retries"] + 1):
            response = self.session.get(self.settings["url"] + path, timeout=self.settings["timeout_seconds"],
                                        headers=headers)
            if response.status_code not in self.settings["retry_statuses"]:
                return response
            if attempt < self.settings["retries"]:
                time.sleep(self.settings["retry_wait_seconds"])
        return response

    def table(self) -> pd.DataFrame:
        paths = [unit.outputs[0] for unit in self.units()]
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} header chunks missing in {self.out_dir}; run fetch-vindr-headers")
        return pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False) for path in paths], ignore_index=True)
