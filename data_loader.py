# from stain_normalizer import WSI_normalizer

from pathlib import Path
import json
import polars as pl
import numpy as np


import openslide
from PIL import Image
import torch
from torchstain import normalizers
import os

from collections import defaultdict
from pprint import pprint

DATA_PATH = Path(r"C:\Users\Eric Arnold\Documents\TCGA_data\tcga_brca_slides")
LOG_PATH = Path(__file__).parent
LOG_PATH.mkdir(exist_ok=True)

REFERENCE_MPP = .25
TILE_SIZE = 256

class WSI_loader:

    def __init__(self, run_name, reference_tile: tuple = (0, 10000)):

        log_path = LOG_PATH / f"{run_name}.jsonl"

        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            self.log = pl.DataFrame(lines)
        else:
            self.log = pl.DataFrame()
    
        self.enumerate_tiles()
        self.reference = self.get_tile(reference_tile)
        # WSI_normalizer()

    def check_finished(self):
        last_entry = self.log[-1]
        self.epoch = last_entry["epoch"]
        completed_tiles = set(self.log.filter(pl.col("epoch") == self.epoch)["tile_id"])


        if len(num_tiles) != len(completed_tiles):
            self.patient = last_patient
            self.to_complete = set(image_files) - set(completed_images)
        else:
            completed_patients = set(self.log["patient"].unique())
            all_patients = set(DATA_PATH.iterdir())
            candidates = all_patients - completed_patients

            if len(candidates) > 0:
                candidate = np.random.choice(candidates)
            else:
                self.epoch += 1
                candidate = np.random.choice(all_patients)

            self.patient = candidate
            candidate_path = DATA_PATH / candidate
            image_path = next(candidate_path.iterdir())
            self.to_complete = set(image_path.iterdir())

    def enumerate_tiles(self):
        patients = list(DATA_PATH.iterdir())
        tile_lookup = defaultdict(dict)
        current_tile = 0
        for p in enumerate(patients):
            img = WSISlide(p)
            res = img.dim
            max_x = res[0]
            max_y = res[1]
            tiles_per_svs = int(max_x * max_y / TILE_SIZE ** 2)
            y_level = -1
            for t in range(tiles_per_svs):
                x_crd = (current_tile * TILE_SIZE) % max_x
                if x_crd == 0:
                    y_level += 1
                y_crd = y_level * TILE_SIZE
                tile_lookup[p.name][current_tile] = (x_crd, y_crd)
                current_tile += 1
            y_level = -1

        self.patients = patients
        self.tile_lookup = tile_lookup
        self.indptr = np.cumsum([len(tile_lookup[key]) for key in tile_lookup])
    

    def fit_normalizer(self, tile):
        reference_tensor = torch.from_numpy(np.array(tile)).permute(2, 0, 1).float()
        normalizer = normalizers.MacenkoNormalizer(backend="torch")
        normalizer.fit(reference_tensor)


    def get_tile(self, tile_tuple):
        patient_path = self.patients[tile_tuple[0]]
        slide = WSISlide(patient_path)
        tile_crds = self.tile_lookup[patient_path.name][tile_tuple[1]]
        tile = slide.read(tile_crds[0], tile_crds[1])
        return tile
    
    # def get_svs(self, patient_path):
    #     x = [f for f in patient_path.iterdir() if f.suffix == ".svs"]
    #     return x[0]
    
    def get_chunk(self, patient_number, x=-1, y=-1, w=1920, h=1080):
        path = self.patients[patient_number]
        slide = WSISlide(path)
        width, height = slide.dim
        
        if x == -1 and y == -1:
            x = (width // 2) - (w // 2)
            y = (height // 2) - (h // 2)
        
        image = slide.read(x, y, w, h)
        image.show()

    def check_empty(self, tile):
        arr = np.array(tile)
        return np.all(arr == 255)
        


class WSISlide:

    def __init__(self, patient_path):
        img_path = [f for f in patient_path.iterdir() if f.suffix == ".svs"]
        self.slide = openslide.OpenSlide(img_path[0])
        self.x = float(self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
        self.y = float(self.slide.properties.get(openslide.PROPERTY_NAME_MPP_Y))
        self.mag = float(self.slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER))
        self.dim = self.slide.level_dimensions[0]
    
    def read(self, x, y, w=TILE_SIZE, h=TILE_SIZE):
        return self.slide.read_region((x, y), 0, (w, h)).convert("RGB")

    

if __name__ == "__main__":
    l = WSI_loader("test")
    l.enumerate_tiles()
    l.get_chunk(2, 0, 0, 10000, 10000)





    