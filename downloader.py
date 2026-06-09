import requests
import os
import json

save_dir = r"C:\Users\Eric Arnold\Documents\TCGA_data\tcga_brca_slides"
os.makedirs(save_dir, exist_ok=True)

# Query GDC API for TCGA-BRCA slide images
params = {
    "filters": json.dumps({
        "op": "=",
        "content": {"field": "project.project_id", "value": "TCGA-BRCA"}
    }),
    "facets": "data_category,data_type,data_format",
    "size": "0"
}

response = requests.get("https://api.gdc.cancer.gov/files", params=params)
data = response.json()
print(data["data"]["pagination"])
files = data["data"]["hits"]
for f in files:
    print(f["file_id"], f["file_name"])