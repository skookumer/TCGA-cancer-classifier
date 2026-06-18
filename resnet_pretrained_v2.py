from pathlib import Path
DATA_PATH = Path(r"C:\Users\Eric Arnold\Documents\TCGA_data")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from clf_head import RESNET
from data_loader import WSI_loader

from sklearn.metrics import f1_score
from tqdm import tqdm

from data_loader import DALIdataset
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

import os
import contextlib


def compute_total_loss(logits, labels, pred_probs, probs, init_prob_loss, init_label_loss):
    prob_loss = BCELoss(pred_probs, probs)
    label_loss = CELoss(logits, labels)
    label_loss = (label_loss * probs).mean()
    return ((prob_loss / init_prob_loss) + (label_loss / init_label_loss))

def unpack_dali(batch):
    tiles       = batch[0]["imgs"].float() / 255.0
    labels = batch[0]["subtypes"].squeeze().long().to(device)
    probs = batch[0]["tumor_probs"].to(device)
    tiles = tiles.float() / 255.0
    return tiles, labels, probs

run_name = "pretrained_2_vprob_l4"
workers  = 16
batch_size = 256

with contextlib.redirect_stderr(open(os.devnull, 'w')):
    loader = DALIdataset(run_name=run_name, downsample=False, batch_size=batch_size, num_workers=workers)
    train_loader = DALIGenericIterator(loader.train, output_map=["imgs", "subtypes", "tumor_probs"], size=len(loader.train_indices), last_batch_policy=LastBatchPolicy.DROP)
    val_loader =   DALIGenericIterator(loader.val,   output_map=["imgs", "subtypes", "tumor_probs"], size=len(loader.val_indices),   last_batch_policy=LastBatchPolicy.DROP)

device = torch.device("cuda:0")
model = RESNET(run_name, unfreeze_l4=True).to(device)
# optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-5)
optimizer = torch.optim.Adam([
    {"params": model.head.parameters(), "lr": 1e-5},
    {"params": model.resnet.layer4.parameters(), "lr": 1e-6},
])

CELoss = nn.CrossEntropyLoss(weight=loader.get_class_weights().to(device),) #reduction="none")
BCELoss = nn.BCELoss()

best_val_acc = 0
patience = 100
patience_counter = 0

# with torch.no_grad():
#     batch = next(iter(train_loader))
#     tiles, labels, probs = unpack_dali(batch)
    
#     logits, pred_probs = model(tiles)
    
#     init_prob_loss  = BCELoss(pred_probs, probs).item()
#     init_label_loss = (CELoss(logits, labels) * probs).mean().item()

init_prob_loss = 1.0
init_label_loss = 1.0

for epoch in tqdm(range(loader.epoch, 100), desc="Epochs"):
    model.train()
    model.resnet.eval()
    if model.unfreeze_l4:
        model.resnet.layer4.train()

    train_loss = 0
    for batch in tqdm(train_loader, desc=f"  Train {epoch}", leave=False):
        tiles, labels, probs = unpack_dali(batch)
        optimizer.zero_grad()

        logits, pred_probs = model(tiles)
        # loss = compute_total_loss(logits, labels, pred_probs, probs, init_prob_loss, init_label_loss)
        loss = CELoss(logits, labels)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(loader.train_indices)

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    # all_probs = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"  Val   {epoch}", leave=False):
            tiles, labels, probs = unpack_dali(batch)

            logits, pred_probs = model(tiles)
            # val_loss += compute_total_loss(logits, labels, pred_probs, probs, init_prob_loss, init_label_loss).item()
            val_loss += CELoss(logits, labels).item()

            preds = logits.argmax(dim=1)
            # correct += ((preds == labels) * probs.squeeze()).sum().item()
            # total   += probs.squeeze().sum().item()
            correct += (preds == labels).sum().item()
            total += len(labels)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            # all_probs.extend(probs.squeeze().cpu().tolist())

    val_loss /= len(loader.val_indices)
    val_acc = correct / total

    per_class_f1 = f1_score(all_labels, all_preds, average=None,) #sample_weight=all_probs)
    class_f1 = {loader.classes[i]: per_class_f1[i] for i in range(len(loader.classes))}

    loader.write_log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }, class_f1)

    print(f"Epoch {epoch} val_acc: {val_acc:.4f} val_loss: {val_loss:.4f}")

    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     model.save()
    #     patience_counter = 0
    # else:
    #     patience_counter += 1
    #     if patience_counter >= patience:
    #         print(f"Early stopping at epoch {epoch}")
    #         break
    model.save()
    with contextlib.redirect_stderr(open(os.devnull, 'w')):
        train_loader = DALIGenericIterator(loader.train, output_map=["imgs", "subtypes", "tumor_probs"], size=len(loader.train_indices), last_batch_policy=LastBatchPolicy.DROP)
        val_loader =   DALIGenericIterator(loader.val,   output_map=["imgs", "subtypes", "tumor_probs"], size=len(loader.val_indices),   last_batch_policy=LastBatchPolicy.DROP)