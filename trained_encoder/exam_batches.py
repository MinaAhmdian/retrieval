import numpy as np
import torch
from torch.utils.data import Dataset

from image_reader import ImageReader
from trained_encoder.exam_inputs import VIEW_COLUMNS
from trained_encoder.fit_data import FitData


class ExamItems(Dataset):

    def __init__(self, data: FitData, transform):
        self.slot_paths = data.slot_paths
        self.views = data.table[VIEW_COLUMNS].to_numpy(dtype=np.int64)
        self.age = data.table["age_id"].to_numpy(dtype=np.int64)
        self.sex = data.table["sex_id"].to_numpy(dtype=np.int64)
        self.context, self.context_mask = data.context, data.context_mask
        self.labels = data.labels
        self.reports = data.reports
        self.report_keys = data.report_keys
        self.transform = transform
        self.reader = ImageReader()

    def __len__(self) -> int:
        return len(self.slot_paths)

    def __getitem__(self, index: int) -> dict:
        distinct, slot_index = [], []
        for path in self.slot_paths[index]:
            if not path:
                slot_index.append(-1)
                continue
            if path not in distinct:
                distinct.append(path)
            slot_index.append(distinct.index(path))
        images = torch.stack([self.transform(self.reader.open(path)) for path in distinct])
        item = {"images": images, "slot_index": torch.tensor(slot_index),
                "view_ids": torch.tensor(self.views[index]), "text_emb": torch.tensor(self.context[index]),
                "text_mask": torch.tensor(bool(self.context_mask[index])), "age_ids": torch.tensor(self.age[index]),
                "sex_ids": torch.tensor(self.sex[index]), "labels": torch.tensor(self.labels[index], dtype=torch.float32)}
        if self.reports is not None:
            item["report"] = torch.tensor(self.reports[index])
            item["report_key"] = self.report_keys[index]
        return item


def collate(items: list[dict]) -> dict:
    offsets = np.cumsum([0] + [len(item["images"]) for item in items[:-1]])
    slot_index = torch.stack([torch.where(item["slot_index"] >= 0, item["slot_index"] + int(offset), -1)
                              for item, offset in zip(items, offsets)])
    batch = {"images": torch.cat([item["images"] for item in items]), "slot_index": slot_index}
    for key in ("view_ids", "text_emb", "text_mask", "age_ids", "sex_ids", "labels", "report"):
        if key in items[0]:
            batch[key] = torch.stack([item[key] for item in items])
    if "report_key" in items[0]:
        batch["report_keys"] = [item["report_key"] for item in items]
    return batch


def place_tokens(tokens: torch.Tensor, slot_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    slot_mask = slot_index >= 0
    image_tokens = tokens.new_zeros((*slot_index.shape, *tokens.shape[1:]))
    image_tokens[slot_mask] = tokens[slot_index[slot_mask]]
    return image_tokens, slot_mask
