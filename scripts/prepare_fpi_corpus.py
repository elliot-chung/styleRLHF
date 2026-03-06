"""
Build corpus from data/corpus/archive.zip.
Expects the zip to contain:
  - images/<id>.jpg
  - styles.csv with columns: id, gender, masterCategory, subCategory, articleType,
    baseColour, season, year, usage, productDisplayName
Maps items to outfit slots (Headwear, Top, Bottom, Shoes, Accessory) and writes catalog.csv.
"""

import csv
from io import TextIOWrapper
import sys
import zipfile
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import IMAGES_DIR, CORPUS_CSV, FPI_ARCHIVE, TOP_SLOT, BOTTOM_SLOT, SHOES_SLOT, ACCESSORY_SLOT


# Map styles.csv (masterCategory, subCategory, articleType) to our slot names
def _map_to_slot(master: str, sub: str, article: str) -> Optional[str]:
    if master == "Footwear":
        return SHOES_SLOT
    elif master == "Accessories":
        return ACCESSORY_SLOT
    elif master == "Apparel":
        if sub == "Topwear":
            return TOP_SLOT
        elif sub == "Bottomwear":
            return BOTTOM_SLOT
        else:
            return ACCESSORY_SLOT
    else: 
        return None

def _map_to_style(usage: str, season: str) -> str:
    temp = "hot" if season == "Summer" or season == "Spring" else "cold" 
    style_table = {
        "Casual": "casual",
        "Sports": "athletic",
        "Ethnic": "casual",
        "Formal": "formal",
        "Smart Casual": "casual",
        "Party": "casual",
        "Travel": "casual",
        "Home": "casual",
    }
    style = style_table.get(usage, "casual")
    return f"{style}_{temp}" if style == "casual" else style

# Final column is the name of the product which can sometimes include commas
# This function reads and fixes the format before opening a csv reader
def _read_csv_bad_format(file_io: TextIOWrapper) -> csv.DictReader:
    file_lines = [ x.decode("utf-8") for x in file_io.readlines() ]
    
    for line_index, original_line in enumerate[str](file_lines):
        line_arr = original_line.split(",")

        if len(line_arr) == 10: continue

        prefix = ",".join(line_arr[:9])
        item_name = f'"{",".join(line_arr[9:]).strip()}"'
        new_line = f"{prefix},{item_name}\n"
        file_lines[line_index] = new_line
    
    return csv.DictReader(file_lines, skipinitialspace=True)

def _construct_catalog_list(reader: csv.DictReader, namelist: list[str]) -> list[dict]:
    output = []
    if not reader.fieldnames:
        print("styles.csv has no header row")
        sys.exit(1)
    for row in reader:
        master = row.get("masterCategory", "")
        sub = row.get("subCategory", "")
        article = row.get("articleType", "")
        item_id = int(row.get("id", ""))
        slot = _map_to_slot(master, sub, article)
        if slot is None:
            continue

        img_name = f"images/{item_id}.jpg"
        if img_name not in namelist:
            continue
        
        usage = row.get("usage", "")
        season = row.get("season", "")
        style_tag = _map_to_style(usage, season)
        output.append({
            "item_id": item_id,
            "category": slot,
            "image_path": img_name,
            "style_tag": style_tag,
        })
    return output    
        
def main():
    if not FPI_ARCHIVE.exists():
        print(f"Not found: {FPI_ARCHIVE}")
        print('Place fashion_product_images(small).zip in "data" folder (containing images/<id>.jpg and styles.csv).')
        sys.exit(1)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(FPI_ARCHIVE, "r") as zf:
        namelist = zf.namelist()
        with zf.open("styles.csv", "r") as f:
            reader = _read_csv_bad_format(f)
            catalog_rows = _construct_catalog_list(reader, namelist)

        # Extract only images we use
        for entry in catalog_rows:
            zip_path = entry["image_path"]
            item_id = entry["item_id"]
            if zip_path:
                dest = IMAGES_DIR / f"{item_id}.jpg"
                if not dest.exists() or dest.stat().st_size == 0:
                    dest.write_bytes(zf.read(zip_path))

    # Write catalog.csv (relative paths under corpus)
    with open(CORPUS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(catalog_rows[0].keys())
        for row in catalog_rows:
            w.writerow(row.values())

    print(f"Wrote {CORPUS_CSV} with {len(catalog_rows)} items.")
    print(f"Images extracted to {IMAGES_DIR}.")


if __name__ == "__main__":
    main()
