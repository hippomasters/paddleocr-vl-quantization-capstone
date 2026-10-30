"""Run directories: server-generated ids and filenames, atomic result writes."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import secrets
import threading

from .uploads import UploadInfo

RUN_ID = re.compile(r"[0-9a-f]{32}")


def valid_run_id(value) -> bool:
    return isinstance(value, str) and RUN_ID.fullmatch(value) is not None


@dataclass(frozen=True)
class Run:
    run_id: str
    directory: Path
    input_path: Path
    info: UploadInfo


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _dir(self, run_id) -> Path:
        if not valid_run_id(run_id):
            raise ValueError("Invalid run id")
        return self.root / run_id

    def create(self, data: bytes, info: UploadInfo) -> Run:
        run_id = secrets.token_hex(16)
        directory = self._dir(run_id)
        directory.mkdir(mode=0o700)
        input_path = directory / f"page{info.extension}"
        with input_path.open("xb") as stream:
            stream.write(data)
        return Run(run_id, directory, input_path, info)

    def exists(self, run_id) -> bool:
        return self._dir(run_id).is_dir()

    def input_file(self, run_id):
        directory = self._dir(run_id)
        for extension in (".png", ".jpg", ".pdf"):
            path = directory / f"page{extension}"
            if path.is_file() and not path.is_symlink():
                return path
        return None

    def write_result(self, run_id, result: dict):
        path = self._dir(run_id) / "result.json"
        temporary = path.with_suffix(".json.tmp")
        with self._lock:
            temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
            os.replace(temporary, path)

    def read_result(self, run_id):
        path = self._dir(run_id) / "result.json"
        with self._lock:
            if not path.is_file():
                return None
            return json.loads(path.read_text(encoding="utf-8"))

    def list_results(self):
        summaries = []
        for entry in self.root.iterdir():
            if not valid_run_id(entry.name) or not entry.is_dir():
                continue
            result = self.read_result(entry.name)
            if result:
                summaries.append(
                    {
                        "run_id": entry.name,
                        "created_utc": result.get("created_utc", ""),
                        "status": result.get("status", ""),
                        "input_kind": result.get("input", {}).get("kind"),
                    }
                )
        return sorted(summaries, key=lambda item: item["created_utc"], reverse=True)
