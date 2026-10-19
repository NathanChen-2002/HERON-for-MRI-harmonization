import os
import json

def generate_nii_json(parent_folder, output_json, exts=(".nii", ".nii.gz")):
    nii_files = dict()

    for root, _, files in os.walk(parent_folder): 
        site_files = dict()
        site_name = os.path.basename(root)
        for f in files:
            if f.lower().endswith(exts):
                abs_path = os.path.abspath(os.path.join(root, f))
                site_files[f] = abs_path
        if site_files: 
            nii_files[site_name] = site_files 


    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(nii_files, f, indent=4, ensure_ascii=False)

    print(f"Found {len(nii_files)} Sites, saved to {output_json}")


##### YOUR CONFIG #####
parent_folder = ""   
output_json = ""
generate_nii_json(parent_folder, output_json)
#######################