"""
Extraction module for Bangladesh DHS Birth Recode (BR) ZIP archives.
Unpacks .dta files into data/raw/ organized by survey round.
"""

import zipfile
from pathlib import Path
from src.utils.config import ZIPS_DATA_DIR, RAW_DATA_DIR, load_config

def extract_all_br_files():
    """Extract .dta file from each round's zip archive into data/raw/{round}."""
    config = load_config()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    extracted_paths = {}

    for item in config["rounds"]:
        rnd = item["round"]
        zip_name = item["zip_name"]
        zip_path = ZIPS_DATA_DIR / zip_name
        target_dir = RAW_DATA_DIR / rnd
        target_dir.mkdir(parents=True, exist_ok=True)

        if not zip_path.exists():
            raise FileNotFoundError(f"Missing zip file for round {rnd} at: {zip_path}")

        print(f"Extracting {rnd} from {zip_name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                if member.lower().endswith(".dta"):
                    # Extract directly into target_dir with standard filename
                    extracted_filename = Path(member).name
                    target_file = target_dir / extracted_filename
                    if not target_file.exists():
                        with zf.open(member) as source, open(target_file, "wb") as dest:
                            dest.write(source.read())
                        print(f"  Extracted: {target_file} ({target_file.stat().st_size / (1024*1024):.2f} MB)")
                    else:
                        print(f"  Already exists: {target_file}")
                    extracted_paths[rnd] = target_file

    print("All BR files extracted successfully.")
    return extracted_paths

if __name__ == "__main__":
    extract_all_br_files()

