import os
import json
import shutil
import cv2
import numpy as np
from shapely.wkt import loads

RAW_IMG_DIR = "tier3/images"
RAW_LABEL_DIR = "tier3/labels"

TARGET_IMG_DIR = "data/images"
TARGET_MASK_DIR = "data/masks"

def parse_wkt_to_array(wkt_string):
    """Converts a geometry string from the JSON into clean pixel points"""
    poly = loads(wkt_string)
    # Extract integer coordinates for OpenCV drawing
    coords = np.array(poly.exterior.coords, dtype=np.int32)
    return coords

def automate_data_pipeline(limit=300):
    # Create clean directories if they don't exist
    os.makedirs(TARGET_IMG_DIR, exist_ok=True)
    os.makedirs(TARGET_MASK_DIR, exist_ok=True)

    print("[Data Engine] Scanning tier3 folder for post-disaster files...")
    all_files = os.listdir(RAW_IMG_DIR)
    # We only care about the post-disaster images to find damage
    post_images = [f for f in all_files if "_post_disaster.png" in f]
    
    post_images = post_images[:limit]
    print(f"[Data Engine] Found {len(post_images)} targets to process. Generating masks...")

    processed_count = 0

    for img_name in post_images:
        base_name = img_name.replace(".png", "")
        json_name = f"{base_name}.json"
        
        img_source_path = os.path.join(RAW_IMG_DIR, img_name)
        json_source_path = os.path.join(RAW_LABEL_DIR, json_name)
        
        if not os.path.exists(json_source_path):
            continue

        mask = np.zeros((1024, 1024), dtype=np.uint8)

        with open(json_source_path, "r") as f:
            label_data = json.load(f)

        has_damage = False
        # Parse through every structural feature in the satellite shot
        for feature in label_data.get("features", {}).get("xy", []):
            wkt_string = feature.get("wkt")
            damage_level = feature.get("properties", {}).get("subtype", "no-damage")

            # We focus purely on highly visible damage zones (Major or Destroyed)
            if damage_level in ["major-damage", "destroyed"]:
                has_damage = True
                try:
                    pixel_coords = parse_wkt_to_array(wkt_string)
                    # Use OpenCV to fill that specific polygon shape with pure white (255)
                    cv2.fillPoly(mask, [pixel_coords], 255)
                except Exception:
                    continue # Skip anomalies or malformed geometry strings

        if has_damage:
            # Save the clean target files with simple matching names
            clean_filename = f"disaster_{processed_count}.png"
            
            shutil.copy(img_source_path, os.path.join(TARGET_IMG_DIR, clean_filename))
            cv2.imwrite(os.path.join(TARGET_MASK_DIR, clean_filename), mask)
            
            processed_count += 1

    print(f"[Success] Data pipeline finished. Compiled {processed_count} paired images and masks into 'data/' folder.")

if __name__ == "__main__":
    automate_data_pipeline(limit=400)
