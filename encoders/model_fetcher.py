import hashlib
import subprocess
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

from config import Config


class ModelFetcher:

    def __init__(self, config: Config):
        self.encoders = config.get("retrieval.encoders")

    def run(self) -> None:
        failed = []
        for name, sources in self.encoders.items():
            try:
                for hub in sources.get("hub", []):
                    snapshot_download(hub["repo"], revision=hub["revision"], allow_patterns=hub["files"])
                for checkout in sources.get("git", []):
                    self.fetch_git(checkout)
                for checkpoint in sources.get("checkpoints", []):
                    self.fetch_checkpoint(checkpoint)
            except Exception as error:
                failed.append(f"{name}: {error}")

        if failed:
            raise RuntimeError("Encoder downloads failed: " + " | ".join(failed))

    @staticmethod
    def fetch_git(checkout: dict) -> None:
        target = Path(checkout["dir"])
        if not target.exists():
            subprocess.run(["git", "clone", "--quiet", checkout["url"], str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--quiet", checkout["commit"]], check=True)
        head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True).stdout.strip()
        if head != checkout["commit"]:
            raise RuntimeError(f"{target} is at {head}, not {checkout['commit']}")

    @staticmethod
    def fetch_checkpoint(checkpoint: dict) -> str:
        path = Path(checkpoint["path"])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".partial")
            urllib.request.urlretrieve(checkpoint["url"], partial)
            partial.rename(path)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
        if checkpoint["sha256"] is not None and digest.hexdigest() != checkpoint["sha256"]:
            raise RuntimeError(f"{path} has sha256 {digest.hexdigest()}, not {checkpoint['sha256']}")
        return digest.hexdigest()
