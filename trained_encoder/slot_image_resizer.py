from config import Config
from encoders.image_resizer import ImageResizer
from provenance import ProvenanceRecords
from trained_encoder.slot_images import ExtraImages


class SlotImageResizer(ImageResizer):

    step = "resize-slot-images"

    def __init__(self, config: Config, corpus_name: str):
        super().__init__(config, corpus_name)
        self.exam_images = ExtraImages(config, corpus_name)
        self.records = ProvenanceRecords(self.exam_images.resized_dir, self.step,
                                         self.step_settings(config, corpus_name))

    def run(self) -> None:
        slot_images = self.exam_images.slot_images
        if len(slot_images.extra()):
            super().run()
