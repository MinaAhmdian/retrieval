import base64
import hashlib
from pathlib import Path

from config import Config


class CheXpertDownload:

    def __init__(self, config: Config):
        self.settings = config.get("corpora.chexpert.redivis")
        self.metadata = config.path("corpora.chexpert.metadata")
        self.labels_dir = config.path("corpora.chexpert.shipped_labels").parent
        self.zips_dir = Path(self.settings["zips_download_dir"])

    def run(self) -> None:
        import redivis

        dataset = redivis.organization(self.settings["organization"]).dataset(
            self.settings["dataset"], version=self.settings["version"])
        if not self.metadata.exists():
            self.metadata.parent.mkdir(parents=True, exist_ok=True)
            dataset.table(self.settings["metadata_table"]).download(path=str(self.metadata), format="csv")
        self.fetch_files(dataset.table(self.settings["labels_table"]), self.labels_dir)
        self.fetch_files(redivis.table(self.settings["images_table"]), self.zips_dir)

    def fetch_files(self, table, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for row in table.to_pandas_dataframe().itertuples(index=False):
            target = out_dir / row.file_name
            if target.exists() and target.stat().st_size == int(row.size) and self.md5_matches(target, row.md5_hash):
                continue
            table.file(row.file_name).download(path=str(out_dir), overwrite=True)
            if target.stat().st_size != int(row.size) or not self.md5_matches(target, row.md5_hash):
                raise RuntimeError(f"{row.file_name}: size or md5 differs from the Redivis listing")

    @staticmethod
    def md5_matches(path: Path, expected: str) -> bool:
        digest = hashlib.md5()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
        return str(expected) in (digest.hexdigest(), base64.b64encode(digest.digest()).decode())
