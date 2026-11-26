import json
import time
from pathlib import Path


class Trace:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a")

    def record(self, row: dict) -> None:
        self.f.write(json.dumps({"ts": time.time(), **row}, default=str) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
