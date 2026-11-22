import io
import os
import zipfile
from pathlib import Path

from PIL import Image

ZIP_SEPARATOR = "::"


class ImageReader:

    def __init__(self):
        self.zips: dict[tuple[int, str], zipfile.ZipFile] = {}

    @staticmethod
    def zip_path(zip_file: str | Path, member: str) -> str:
        return f"{zip_file}{ZIP_SEPARATOR}{member}"

    def zip_file(self, path: str) -> zipfile.ZipFile:
        key = (os.getpid(), path)
        if key not in self.zips:
            self.zips[key] = zipfile.ZipFile(path)
        return self.zips[key]

    def open(self, image_path: str) -> Image.Image:
        if ZIP_SEPARATOR in image_path:
            zip_file, member = image_path.split(ZIP_SEPARATOR, 1)
            source = io.BytesIO(self.zip_file(zip_file).read(member))
        else:
            source = image_path
        with Image.open(source) as image:
            image.load()
        return image

    def exists(self, image_path: str) -> bool:
        if ZIP_SEPARATOR not in image_path:
            return Path(image_path).is_file()
        zip_file, member = image_path.split(ZIP_SEPARATOR, 1)
        if not Path(zip_file).is_file():
            return False
        return member in self.zip_file(zip_file).NameToInfo
