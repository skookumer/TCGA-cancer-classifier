import requests
import os
import json
import polars as pl

save_dir = r"C:\Users\Eric Arnold\Documents\TCGA_data"
os.makedirs(save_dir, exist_ok=True)

for f in os.listdir(save_dir):
    print(f)

full = pl.read_csv(os.path.join(save_dir, "gdc_manifest.tsv.txt"), separator="\t")
already_have = pl.read_csv(os.path.join(save_dir, "gdc_manifest_20.tsv"), separator="\t")

subset = full.head(200)
new_only = subset.filter(~pl.col("id").is_in(already_have["id"]))

print(f"Total in subset: {len(subset)}, already have: {len(already_have)}, new to download: {len(new_only)}")

new_only.write_csv(os.path.join(save_dir, "manifest_next.tsv"), separator="\t")