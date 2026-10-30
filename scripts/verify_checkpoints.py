"""Verify published model files against the benchmark's recorded SHA-256 values."""

from pathlib import Path
import hashlib


def verify_checkpoint(directory: Path, expected: dict) -> dict:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    executable_suffixes = {".py", ".pyc", ".pth", ".so", ".dll", ".dylib"}
    unexpected = sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.suffix.lower() in executable_suffixes and entry.name not in expected
    )
    if unexpected:
        raise ValueError(f"Unlisted executable files: {unexpected}")
    verified = []
    for name, record in sorted(expected.items()):
        if Path(name).name != name:
            raise ValueError(f"Unsafe checkpoint filename: {name!r}")
        path = directory / name
        if path.is_symlink():
            raise ValueError(f"Symlinked checkpoint file: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Size mismatch: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {name}")
        verified.append(name)
    return {"verified_files": verified}


def main() -> int:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", required=True, choices=["BF16", "W8A16", "W4A16"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    try:
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
        expected = data["variants"][args.variant]["expected_files"]
        result = verify_checkpoint(args.checkpoint, expected)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"variant": args.variant, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
