import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pandas as pd

from corpora.base import IMAGE_COLUMNS, REPORT_COLUMNS, Corpus


class OpenICorpus(Corpus):

    name = "openi"

    def __init__(self, config):
        super().__init__(config)
        self.reports_by_id = [self.read_report(path) for path in sorted(Path(self.setting("reports_dir")).glob("*.xml"))]

    def read_report(self, path: Path) -> dict:
        root = ElementTree.parse(path).getroot()
        sections = {}
        for element in root.iter("AbstractText"):
            sections[element.get("Label")] = " ".join((element.text or "").split())
        return {
            "report_id": root.find("uId").get("id"),
            "sections": sections,
            "image_ids": [element.get("id") for element in root.iter("parentImage")],
        }

    def images(self) -> pd.DataFrame:
        images_dir = Path(self.setting("images_dir"))
        rows = [{
            "exam_id": report["report_id"],
            "patient_id": report["report_id"],
            "image_id": image_id,
            "image_path": str(images_dir / f"{image_id}.png"),
            "view_class": "unknown",
            "view_label": "unknown",
            "split": "",
            "age": float("nan"),
            "sex": "",
        } for report in self.reports_by_id for image_id in report["image_ids"]]
        return self.check_columns(pd.DataFrame(rows), IMAGE_COLUMNS)

    def reports(self) -> pd.DataFrame:
        label_sections = self.setting("label_sections")
        rows = [{
            "exam_id": report["report_id"],
            "label_text": " ".join(report["sections"][name] for name in label_sections if report["sections"].get(name)),
            "context_text": "",
        } for report in self.reports_by_id]
        return self.check_columns(pd.DataFrame(rows), REPORT_COLUMNS)
