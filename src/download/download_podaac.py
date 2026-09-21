"""Download configured PO.DAAC collections with earthaccess."""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


def download_all(config_path: str | Path = "config.yaml") -> None:
    """Authenticate and download every configured PO.DAAC product."""
    import earthaccess

    config = load_config(config_path)
    domain = config["domain"]
    dates = config["dates"]
    output_root = Path(config["paths"]["podaac_output"])
    output_root.mkdir(parents=True, exist_ok=True)

    earthaccess.login(persist=True)
    for product in config["podaac"]["products"].values():
        output_directory = output_root / product["output_subdirectory"]
        output_directory.mkdir(parents=True, exist_ok=True)
        granules = earthaccess.search_data(
            short_name=product["short_name"],
            version=product["version"],
            temporal=(dates["start"], dates["end"]),
            bounding_box=(domain["west"], domain["south"], domain["east"], domain["north"]),
        )
        earthaccess.download(granules, local_path=str(output_directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    download_all(parser.parse_args().config)
