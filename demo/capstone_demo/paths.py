"""Fixed locations of the tested reproduction scripts this demo delegates to."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEMO_DIR = PACKAGE_DIR.parent
SOURCE_DIR = DEMO_DIR.parent
SOURCE_SCRIPTS = SOURCE_DIR / "scripts"
SERVE_SCRIPT = SOURCE_SCRIPTS / "serve_model.py"
PARSE_SCRIPT = SOURCE_SCRIPTS / "parse_page.py"
MANIFEST = SOURCE_DIR / "config" / "checkpoint_hashes.json"
CLIENT_SCRIPT = PACKAGE_DIR / "timed_parse.py"
STATIC_DIR = PACKAGE_DIR / "static"
