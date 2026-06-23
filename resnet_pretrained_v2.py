from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from clf_head import RESNET, INCEPTION, VGG16, AGG_14, RESNET_CUSTOM
from data_loader import WSI_loader

from sklearn.metrics import f1_score
from tqdm import tqdm

from data_loader import DALIdataset, unpack_dali
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


def compute_total_loss(logits, labels, pred_probs, probs, init_prob_loss, init_label_loss):
    prob_loss = BCELoss(pred_probs, probs)
    label_loss = CELoss(logits, labels)
    label_loss = (label_loss * probs).mean()
    return ((prob_loss / init_prob_loss) + (label_loss / init_label_loss))

run_name = "r18-d50-v3"
workers  = 16
batch_size = 256


loader       = DALIdataset(run_name=run_name, downsample=False, batch_size=batch_size, num_workers=workers)
train_loader = DALIGenericIterator(loader.train, output_map=["imgs", "subtypes", "tumor_probs"])
val_loader   = DALIGenericIterator(loader.val,   output_map=["imgs", "subtypes", "tumor_probs"])

device = torch.device("cuda:0")
model = RESNET(run_name, unfreeze_last=True, dropout=0.5).to(device)
# optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-5)

optimizer = torch.optim.Adam([
    {"params": model.head.parameters(), "lr": 1e-5},
    # {"params": model.last.parameters(), "lr": 1e-5},
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

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

train_steps = len(loader.train_indices) // batch_size
val_steps = len(loader.val_indices) // batch_size
for epoch in tqdm(range(loader.epoch, 20), desc="Epochs"):
    model.train()
    model.pretrained.eval()
    # if model.unfreeze_last:
    #     model.last.train()

    train_loss = 0
    for batch in tqdm(train_loader, total=train_steps, desc=f"  Train {epoch}", leave=False):
        tiles, labels, probs = unpack_dali(batch, device)
        optimizer.zero_grad()

        logits = model(tiles)
        # loss = compute_total_loss(logits, labels, pred_probs, probs, init_prob_loss, init_label_loss)
        loss = CELoss(logits, labels)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= train_steps

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    # all_probs = []
    with torch.no_grad():
        for batch in tqdm(val_loader, total=val_steps, desc=f"  Val {epoch}", leave=False):
            tiles, labels, probs = unpack_dali(batch, device)

            logits = model(tiles)
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

    val_loss /= val_steps
    val_acc = correct / total

    per_class_f1 = f1_score(all_labels, all_preds, average=None,) #sample_weight=all_probs)
    class_f1 = {loader.classes[i]: per_class_f1[i] for i in range(len(loader.classes))}

    loader.write_log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        **class_f1
    })

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
    scheduler.step(val_loss)

    train_loader = DALIGenericIterator(loader.train, output_map=["imgs", "subtypes", "tumor_probs"])
    val_loader =   DALIGenericIterator(loader.val,   output_map=["imgs", "subtypes", "tumor_probs"])