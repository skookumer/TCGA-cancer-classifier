# from stain_normalizer import WSI_normalizer

from pathlib import Path
import json
import polars as pl
import pandas as pd
import numpy as np

import sys
import platform


import openslide
from PIL import Image
import torch
from torchstain import normalizers
from torchvision import transforms
from torch.utils.data import Dataset
import os

from collections import OrderedDict
from pprint import pprint
from tqdm import tqdm

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

if platform.system() == "Linux":
    from nvidia.dali import pipeline_def, fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    import nvidia.dali.types as types
    
# if platform.system() == "Linux":
#     DATA_PATH = Path("/mnt/c/Users/Eric Arnold/Documents/TCGA_data/tcga_brca_slides")
#     JPG_PATH = Path("/home/ruminator/tcga_brca_jpgs")
# else:
#     DATA_PATH = Path(r"C:\Users\Eric Arnold\Documents\TCGA_data\tcga_brca_slides")
#     JPG_PATH = DATA_PATH.parent / "tcga_brca_jpgs"

DATA_PATH = Path("/home/eric/TCGA_data/tcga_brca_slides")
JPG_PATH = DATA_PATH.parent / "tcga_brca_jpgs"

DATA_PATH.mkdir(exist_ok=True)
JPG_PATH.mkdir(exist_ok=True)

LOG_PATH = Path(__file__).parent / "logs"
LOG_PATH.mkdir(exist_ok=True)

DF_PATH = Path(__file__).parent / "tile_lookup_clf.parquet"
REFERENCE_MPP = .25
TILE_SIZE = 512
ACCOUNT_FOR_MPP = False
LABEL_MAP = {"LumA": 0, "LumB": 1, "Basal": 2, "Her2": 3}

class WSI_loader:

    def __init__(self, run_name="none", cache_max=24, downsample=False, tumor_prob=.9, balance_classes=True):
        '''
        Data loading class that has multiple ways of serving up the data. Development went like this:

        read from WSIs (openslide) -> read from jpeg (PIL) -> subset w/ tumor_prob -> Nvidia DALI
        (CPU decode)                  (CPU decode, transform GPU)                     (all GPU)

        So there are multiple redundant aspects to this class, but I kept it for posterity.
        But CPU decoding can be used to maximize GPU througput with a high-powered system.
        '''

        self.run_name = run_name
        self.log_path = LOG_PATH / f"{run_name}.jsonl"
        self.downsample = downsample
        self.tumor_prob = tumor_prob
        self.balance_classes = balance_classes

        if self.log_path.exists():
            with open(self.log_path, "r", encoding="utf-8") as f:
                last_line = json.loads(f.readlines()[-1])
                self.epoch = last_line["epoch"] + 1
        else:
            self.epoch = 0

                
        self.log_cols = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "per_class_f1", "lr"]
        self.classes = {v: k for k, v in LABEL_MAP.items()}

        self.patient_lookup = {p.name: p for p in list(DATA_PATH.iterdir())}

        #these are cpu normalizations for the initial version of the program
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=360, fill=255),
            transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.05),
        ])

        self.to_tensor = transforms.ToTensor()

        self.enumerate_tiles()
        self.get_labels()
        self.set_indices()
        self.slide_cache = {} #the original approach of reading data directly from the WSIs using openslide (ram-hungry)
        self.cache_max = cache_max

    def enumerate_tiles(self):
        '''
        Creates a manifest of all tiles and their coordinates for retrieval directly from WSIs.
        If new data are downloaded from GDC, this function segments WSIs into tiles and checks
        to see if they're mostly empty.

        Uses parallelization for speed.
        '''
        cols = {
            "patient_name": pl.Utf8,
            "crd": pl.List(pl.Int64),
            "is_empty": pl.Boolean
        }

        #.with_row_index("idx")
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

            return rows

        if len(patients) > 0:
            results = Parallel(n_jobs=4, return_as="generator_unordered")(
                delayed(process_patient)(p) for p in patients
            )

            for rows in results:
                df_new = pl.DataFrame(rows, schema=cols)
                self.df = pl.concat([self.df, df_new])
                self.df.write_parquet(DF_PATH)

    def get_labels(self):
        '''
        Uses the metadata from GDC to get the folder names and align them with the entity_ids
        (and other ids) from the UCSC Xena clinical matrix.
        '''
        with open(DATA_PATH.parent / "metadata.repository.2026-06-04.json") as f:
            metadata = json.load(f)

        slides = []
        for entry in metadata:
            for entity in entry["associated_entities"]:
                slides.append({
                    "patient_name": entity["entity_submitter_id"],
                    "sample_id": entity["entity_submitter_id"][:15],
                    "entity_id": entity["entity_id"],
                    "file_id": entry["file_id"],
                    "file_name": entry["file_name"]
                })

        slides_df = pd.DataFrame(slides)

        clinical = pd.read_csv(DATA_PATH.parent / "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix", sep="\t")
        pam50 = clinical[["sampleID", "PAM50Call_RNAseq"]].dropna()
        pam50 = pam50[pam50["PAM50Call_RNAseq"] != "Normal"]

        result = slides_df.merge(pam50, left_on="sample_id", right_on="sampleID", how="inner")
        result = result[["patient_name", "entity_id", "file_id", "PAM50Call_RNAseq"]].rename(columns={"PAM50Call_RNAseq": "subtype"})

        self.label_lookup = pl.from_pandas(result)

        self.df = self.df.join(
            self.label_lookup.select(["file_id", "subtype"]),
            left_on="patient_name",   # UUID in self.df
            right_on="file_id",       # matching UUID in label_lookup
            how="left"
        )

    def set_indices(self):
        '''
        Aligns the preprocessed indices of individual tiles with molecular subtypes.
        Facilitates train/val/test split and sets separate indices as attributes.
        Splits the dataset on patient to prevent data leakage.     
        '''

        index_df = (
            self.df
            .with_row_index()
            .filter(~pl.col("is_empty"))
            .select(["index", "patient_name", "tumor_prob"])
            .join(
                self.label_lookup.select(["file_id", "subtype"]),
                left_on="patient_name",   # UUID folder name
                right_on="file_id",       # matching UUID from metadata
                how="inner"
            )
        )

        if self.balance_classes:
            subtypes = [index_df.filter(pl.col("subtype") == s) for s in LABEL_MAP]
            min_count = min([df.shape[0] for df in subtypes])
            subtypes = [df.sort("tumor_prob", descending=True)[:min_count] for df in subtypes]
            index_df = pl.concat(subtypes)

        patients = index_df["patient_name"].unique().to_list()
        
        if self.downsample:
            patients = patients[:self.downsample]

        labels = [index_df.filter(pl.col("patient_name") == p)["subtype"][0] for p in patients]

        # split off test
        train_patients, test_patients, train_labels, _ = train_test_split(patients, labels, stratify=labels, test_size=0.2, random_state=42)

        # split train into train/val
        train_patients, val_patients = train_test_split(train_patients, stratify=train_labels, test_size=0.2, random_state=42)

        self.train_indices = index_df.filter(pl.col("patient_name").is_in(train_patients))["index"].to_list()
        self.val_indices   = index_df.filter(pl.col("patient_name").is_in(val_patients))["index"].to_list()
        self.test_indices  = index_df.filter(pl.col("patient_name").is_in(test_patients))["index"].to_list()
        self.train_df = index_df.filter(pl.col("patient_name").is_in(train_patients))
    
    def write_jpgs(self):
        '''
        Instead of opening WSIs (openslide) from disk or from cache, decided to write non-empty tiles as
        resized jpegs to a folder for quicker and more efficient retrieval with PIL.
        '''

        non_empty = self.df.filter(~pl.col("is_empty"))
        patients = non_empty["patient_name"].unique().to_list()
        subsets = [non_empty.filter(pl.col("patient_name") == name) for name in patients]

        def get_and_write_downsized_tile(subset):
            p = subset["patient_name"][0]
            img = WSISlide(DATA_PATH / p)
            for row in subset.iter_rows(named=True):
                crds = row["crd"]
                idx = row["idx"]
                tile = img.read(crds[0], crds[1])
                tile = tile.resize((224, 224), Image.LANCZOS)
                tile.save(JPG_PATH / f"{idx}.jpg", quality=90)
            img.close()

        Parallel(n_jobs=4)(
            delayed(get_and_write_downsized_tile)(s)
            for s in tqdm(subsets, total=len(subsets))
        )

    def get_class_weights(self):
        '''function for serving class weights to the cross entropy loss object'''
        labels = torch.tensor([LABEL_MAP[s] for s in self.train_df["subtype"].to_list()], dtype=torch.long)
        counts = torch.bincount(labels, minlength=4).float()
        weights = 1.0 / counts
        return weights / weights.sum()

    def get_normed_tensor(self, idx, resize_to=224, transform=True):
        '''old function for getting tiles directly from WSIs'''
        tile = self.get_tile(idx)
        if resize_to is not None:
            tile = tile.resize((resize_to, resize_to), Image.LANCZOS)
        if transform:
            tile = self.transform(tile)
        return self.normalize(tile)
    
    def get_tensor(self, idx):
        '''newer function that just gets a tensor for cpu decoding and gpu transforms'''
        return self.to_tensor(self.get_tile(idx))
    
    def get_slide(self, file_id):
        '''old function that puts slides in the slide cache for faster opening per worker'''
        if file_id not in self.slide_cache:
            self.slide_cache[file_id] = WSISlide(self.patient_lookup[file_id])
        return self.slide_cache[file_id]

    def get_tile(self, idx):
        '''re-used function that can load from jpegs or from WSI for CPU transforming/ decoding'''
        jpg_path = JPG_PATH / f"{idx}.jpg"
        if jpg_path.exists():
            return Image.open(jpg_path)
        
        row = self.df[idx]
        file_id = row["patient_name"][0]
        slide = self.get_slide(file_id)
        crds = row["crd"][0]
        return slide.read(crds[0], crds[1])
    
    def get_chunk(self, patient_id, x=-1, y=-1, w=1920, h=1080):
        '''test function that can just get a chunk of a WSI for visualization'''
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

    def get_label(self, subtype):
        '''for pytorch dataloader'''
        return torch.tensor(LABEL_MAP[subtype], dtype=torch.long)

    def __getitem__(self, i):
        '''old method to enable compatibility with the pytorch dataloader class'''
        idx = self.valid_indices[i]
        row = self.df[idx]
        label = self.get_label(row["patient_name"][0])
        tile = self.get_normed_tensor(idx)
        return tile, label
    
    '''properties necessary to allow pytorch to access different tile sets'''

    @property
    def train(self):
        return TileSubset(self, self.train_indices)

    @property
    def val(self):
        return TileSubset(self, self.val_indices)

    @property
    def test(self):
        return TileSubset(self, self.test_indices)
    
    @property
    def all_indices_for_tumor_clf(self):
        return TumorSet(self, self.df.filter(~pl.col("is_empty"))["idx"].to_list())
    
    def write_log(self, log_dict):
        '''log write function to keep pathing self-contained'''
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_dict) + "\n")
    
    def map_tiles(self, patient_name):
        '''function to eventually be used with whole-slide classification'''
        rows = self.df.filter(pl.col("patient_name") == patient_name)
        indices = rows["idx"].to_list()
        coords = [tuple(crd / 512 for crd in crds) for crds in rows["crds"].to_list()]
        jpgs = [Image.open(JPG_PATH / f"{idx}.jpg") for idx in indices]
    
    def get_tile_for_tumor_detection(self, idx):
        '''function to take a 350x350 chunk out of each 512x512 tile for tumor classification'''
        row = self.df[idx]
        file_id = row["patient_name"][0]
        slide = self.get_slide(file_id)
        crds = row["crd"][0]
        offset = (slide.read_size - 350) // 2  # centers the crop
        return slide.read(
            crds[0] + offset,
            crds[1] + offset,
            w_in=350,
            h_in=350,
            w_out=224,
            h_out=224
        )
        

class WSISlide:

    def __init__(self, patient_path):
        '''class to facilitate easy input from WSI slides and to bake in dimensioning'''
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
    
    def close(self):
        self.slide.close()


class TileSubset(Dataset):

    def __init__(self, parent, indices):
        '''child class that allows pytorch to access different index sets'''
        self.parent = parent
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        row = self.parent.df[idx]
        label = self.parent.get_label(row["subtype"][0])
        # tile = self.parent.get_normed_tensor(idx, resize_to=None, transform=False)
        tile = self.parent.get_tensor(idx)
        return tile, label
    
class TumorSet(Dataset):

    def __init__(self, parent, indices):
        '''specific class for the tumor detection script'''
        self.parent = parent
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        idx = self.indices[i]
        tile = self.parent.get_tile_for_tumor_detection(idx)
        return self.parent.to_tensor(tile), idx

if platform.system() == "Linux":
    '''DALI only works on Linux'''

    # @pipeline_def
    '''old pipeline def that uses non-parallel make_source kept as reference'''
    # def dali_pipeline(source):
    #     imgs, subtypes, tumor_probs = fn.external_source(
    #         source=source,
    #         num_outputs=3,
    #         dtype=[types.UINT8, types.INT64, types.FLOAT],
    #         batch=True
    #     )

    #     images = fn.decoders.image(imgs, device="mixed", output_type=types.RGB)
    #     images = fn.transpose(images, perm=[2, 0, 1])  # HWC -> CHW for PyTorch

    #     return images, subtypes, tumor_probs

    @pipeline_def
    def dali_pipeline(file_paths, labels, tumor_probs):
        '''separate function to define the pipeline'''
        imgs, subtypes = fn.readers.file(files=file_paths, labels=labels, random_shuffle=True)
        images = fn.decoders.image(imgs, device="mixed", output_type=types.RGB)
        images = fn.transpose(images, perm=[2, 0, 1])
        probs = fn.external_source(source=tumor_probs, batch=False)
        return images, subtypes, probs

    class DALIdataset(WSI_loader):
        
        def __init__(self, run_name="none", downsample=False, batch_size=32, num_workers=8, device_id=0):
            '''class that returns the properly configured DALI loader'''
            super().__init__(run_name=run_name, downsample=downsample)
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.device_id = device_id 

        def make_source(self, indices):
            '''old method that formats information in a manner that DALI expects
            Images are stored in binary and passed to the GPU to be decoded, hence frombuffer.
            '''
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch = self.df[batch_indices]

                img_indices = batch["idx"].to_list()
                tumor_probs = batch["tumor_prob"].to_list()
                subtypes =    batch["subtype"].to_list()

                imgs =        [np.frombuffer(open(JPG_PATH / f"{idx}.jpg", "rb").read(), dtype=np.uint8) for idx in img_indices]
                subtypes    = [np.array(LABEL_MAP[s], dtype=np.int64) for s in subtypes]
                tumor_probs = [np.array(p,             dtype=np.float32) for p in tumor_probs]

                yield imgs, subtypes, tumor_probs

        def prep_pipeline(self, indices):
            '''
            New pipeline preparation method that works with the new pipeline defeinition (fn.readers)
            '''
            batch = self.df[indices]
            file_paths  = [str(JPG_PATH / f"{idx}.jpg") for idx in batch["idx"].to_list()]
            labels      = batch["subtype"].replace(LABEL_MAP).to_list()
            tumor_probs = np.array(batch["tumor_prob"].to_list(), dtype=np.float32)
            
            pipeline = dali_pipeline(
                file_paths=file_paths,
                labels=labels,
                tumor_probs=tumor_probs,
                batch_size=self.batch_size,
                num_threads=self.num_workers,
                device_id=self.device_id
            )
            pipeline.build()
            return pipeline

        '''old pipeline functions to return the child class'''
        # @property
        # def train(self):
        #     source = self.make_source(self.train_indices)
        #     pipeline = dali_pipeline(source, batch_size=self.batch_size, num_threads=self.num_workers, device_id=self.device_id)
        #     pipeline.build()
        #     return pipeline

        # @property
        # def val(self):
        #     source = self.make_source(self.val_indices)
        #     pipeline = dali_pipeline(source, batch_size=self.batch_size, num_threads=self.num_workers, device_id=self.device_id)
        #     pipeline.build()
        #     return pipeline
        
        # @property
        # def test(self):
        #     source = self.make_source(self.test_indices)
        #     pipeline = dali_pipeline(source, batch_size=self.batch_size, num_threads=self.num_workers, device_id=self.device_id)
        #     pipeline.build()
        #     return pipeline

        '''new pipeline functions'''
        @property
        def train(self):
            return self.prep_pipeline(self.train_indices)
        
        @property
        def val(self):
            return self.prep_pipeline(self.val_indices)
        
        @property
        def test(self):
            return self.prep_pipeline(self.test_indices)

def unpack_dali(batch, device):
    '''functions to unpack DALI batches served by the iterator'''
    tiles  = batch[0]["imgs"].float() / 255.0
    labels = batch[0]["subtypes"].squeeze().long().to(device)
    probs  = batch[0]["tumor_probs"].to(device)
    return tiles, labels, probs


if __name__ == "__main__":
    l = WSI_loader("clf1")
    print(torch.cuda.is_available())       # True or False
    print(torch.version.cuda)              # CUDA version PyTorch was built with
    print(torch.cuda.get_device_name(0))   # GPU name, e.g. "NVIDIA GeForce RTX 3090"
    # print(l.df.head())
    # l.write_jpgs()
    # l.get_chunk(0, w=10000, h=10000)





    