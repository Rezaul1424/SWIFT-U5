"""
Configuration utilities for SWIFT-U5 study.
Loads project parameters, survey round paths, variable mappings, and study settings.
"""

from pathlib import Path
import yaml

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = WORKSPACE_DIR / "configs" / "config.yaml"

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load yaml configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Common constants
RAW_DATA_DIR = WORKSPACE_DIR / "data" / "raw"
ZIPS_DATA_DIR = WORKSPACE_DIR / "data" / "zips"
INTERIM_DATA_DIR = WORKSPACE_DIR / "data" / "interim"
PROCESSED_DATA_DIR = WORKSPACE_DIR / "data" / "processed"
TABLES_DIR = WORKSPACE_DIR / "outputs" / "tables"
FIGURES_DIR = WORKSPACE_DIR / "outputs" / "figures"
REPORTS_DIR = WORKSPACE_DIR / "outputs" / "reports"

