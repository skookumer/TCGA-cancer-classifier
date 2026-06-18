from data_loader import WSI_loader
import torch
from huggingface_hub import hf_hub_download
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import polars as pl

model_path = hf_hub_download(
    repo_id="kaczmarj/breast-tumor-resnet34.tcga-brca",
    filename="torchscript_model.pt"
)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])


l = WSI_loader()
d = torch.device("cuda:0")

model = torch.jit.load(model_path).eval().cuda().to(d)

if __name__ == "__main__":
    df_clf = l.df
    dataset = l.all_indices_for_tumor_clf
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True, prefetch_factor=2)
    probs = np.zeros(df_clf.shape[0])
    print(df_clf.head())

    for batch_imgs, batch_indices in tqdm(dataloader):
        batch_imgs = batch_imgs.cuda(non_blocking=True)
        batch_imgs = normalize(batch_imgs)
        with torch.inference_mode():
            logits = model(batch_imgs)
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        for idx, prob in zip(batch_indices.tolist(), batch_probs):
            probs[idx] = prob
        
    df_clf = df_clf.with_columns(pl.Series("tumor_prob", probs))
    df_clf.write_parquet("tile_lookup_clf.parquet")