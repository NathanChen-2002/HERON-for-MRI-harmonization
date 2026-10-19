import json
import os
import collections

import numpy as np

##### YOUR CONFIG #####
SAVE_PATH = ""

# Set to "train" or "test".
DATA_SPLIT = "train"
#######################

in_json = os.path.join(SAVE_PATH, "datalist.json")
out_json = os.path.join(SAVE_PATH, "preprocessed_datalist.json")

with open(in_json, "r", encoding="utf-8") as f:
    nested = json.load(f)

out = []
for site_list in nested:
    for sample in site_list:
        sample = dict(sample)
        # For test data, remove site markers from sample["subject"] here
        # according to your naming convention.
        sample["datasplit"] = DATA_SPLIT
        out.append(sample)

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=4, ensure_ascii=False)

print(f"Saved {len(out)} {DATA_SPLIT} slices to {out_json}")