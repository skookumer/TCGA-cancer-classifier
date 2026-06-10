# from stain_normalizer import WSI_normalizer

from pathlib import Path
import json
import polars as pl
import numpy as np


import openslide
from PIL import Image
import torch
from torchstain import normalizers
from torchvision import transforms
import os

from collections import defaultdict
from pprint import pprint
from tqdm import tqdm

from joblib import Parallel, delayed

DATA_PATH = Path(r"C:\Users\Eric Arnold\Documents\TCGA_data\tcga_brca_slides")
LOG_PATH = Path(__file__).parent / "logs"
LOG_PATH.mkdir(exist_ok=True)

DF_PATH = Path(__file__).parent / "tile_lookup.parquet"
REFERENCE_MPP = .25
TILE_SIZE = 512
ACCOUNT_FOR_MPP = False

class WSI_loader:

    def __init__(self, run_name):

        log_path = LOG_PATH / f"{run_name}.jsonl"

        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            self.log = pl.DataFrame(lines)
        else:
            self.log = pl.DataFrame()
        
        self.enumerate_tiles()
        self.patient_lookup = {p.name: p for p in list(DATA_PATH.iterdir())}
        self.normalizer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        self.valid_indices = self.df.filter(~pl.col("is_empty"))

    def enumerate_tiles(self):
        cols = {
            "patient_name": pl.Utf8,
            "crd": pl.List(pl.Int64),
            "is_empty": pl.Boolean
        }

        self.df = pl.read_parquet(DF_PATH) if DF_PATH.exists() else pl.DataFrame(schema=cols)
        processed = set(self.df["patient_name"].unique().to_list())
        patients = [p for p in list(DATA_PATH.iterdir()) if p.name not in processed]

        def process_patient(p):
            print(f"enumerating {p.name}")
            img = WSISlide(p)
            res = img.dim
            max_x = res[0]
            max_y = res[1]
            tiles_per_svs = int(max_x * max_y / img.read_size ** 2)

            x_old = 0
            y_level = 0
            current_tile = 0

            rows = []

            for t in tqdm(range(tiles_per_svs)):
                row = {"patient_name": p.name, "crd": None, "is_empty": False}

                x_crd = (current_tile * img.read_size) % max_x
                if x_crd < x_old:
                    y_level += 1
                y_crd = y_level * img.read_size
                
                row["crd"] = (x_crd, y_crd)
                tile = img.read_raw(x_crd, y_crd)
                arr = np.array(tile)
                gray = 0.2989 * arr[:,:,0] + 0.5870 * arr[:,:,1] + 0.1140 * arr[:,:,2]
                row["is_empty"] = bool((gray > 220).mean() > 0.8)

                current_tile += 1
                x_old = x_crd
                rows.append(row)

            y_level = 0
            current_tile = 0
            return rows
        
        if len(patients) > 0:
            results = Parallel(n_jobs=12)(delayed(process_patient)(p) for p in self.patients)
            all_rows = [row for patient_rows in results for row in patient_rows]
            df_new = pl.DataFrame(all_rows)
            self.df = pl.concat([self.df, df_new])
            self.df.write_parquet(DF_PATH)

    def get_normed_tensor(self, idx, resize_to=256):
        tile = self.get_tile(idx)
        if resize_to is not None:
            tile = tile.resize((resize_to, resize_to))
        return self.normalizer(tile)

    def get_tile(self, idx):
        row = self.df[idx]
        patient_path = self.patient_lookup[row["patient_name"]]
        slide = WSISlide(patient_path)
        crds = row["crd"]
        return slide.read(crds[0], crds[1])
    
    # def get_svs(self, patient_path):
    #     x = [f for f in patient_path.iterdir() if f.suffix == ".svs"]
    #     return x[0]
    
    def get_chunk(self, patient_id, x=-1, y=-1, w=1920, h=1080):
        if type(patient_id) != str:
            patient_id = list(self.patient_lookup.keys())[patient_id]
        path = self.patient_lookup[patient_id]
        slide = WSISlide(path)
        width, height = slide.dim

        
        if x == -1 and y == -1:
            x = (width // 2) - (w // 2)
            y = (height // 2) - (h // 2)
        
        aspect = slide.read_size // TILE_SIZE
        
        image = slide.read(x, y, round(aspect * w), round(aspect * h), w, h)
        image.show()
        


class WSISlide:

    def __init__(self, patient_path):
        img_path = [f for f in patient_path.iterdir() if f.suffix == ".svs"]
        self.slide = openslide.OpenSlide(img_path[0])
        self.x = float(self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
        self.y = float(self.slide.properties.get(openslide.PROPERTY_NAME_MPP_Y))
        self.mag = float(self.slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER))
        self.dim = self.slide.level_dimensions[0]
        if ACCOUNT_FOR_MPP:
            self.read_size = int(TILE_SIZE * (self.x / REFERENCE_MPP))
        else:
            self.read_size = TILE_SIZE

    def read(self, x, y, w_in=-1, h_in=-1, w_out=TILE_SIZE, h_out=TILE_SIZE):
        w_in = self.read_size if w_in == -1 else w_in
        h_in = self.read_size if h_in == -1 else h_in
        return self.slide.read_region((x, y), 0, size=(w_in, h_in)).convert("RGB").resize((w_out, h_out))

    def read_raw(self, x, y):
        return self.slide.read_region((x, y), 0, size=(self.read_size, self.read_size)).convert("RGB")

if __name__ == "__main__":
    l = WSI_loader("test")
    l.get_chunk(0, w=10000, h=10000)





    