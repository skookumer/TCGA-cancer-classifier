import torch
import torch.nn as nn
import torch.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

from CNNs import Agglomerator, IMG_Transformer, SupConLoss, Encoder_18
from data_loader import DALIdataset, unpack_dali
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from einops import rearrange
# from pytorch_metric_learning.losses import SupConLoss

def per_class_loss(criterion, output, labels):
    class_losses = {}
    for cls in labels.unique():
        mask = labels == cls
        if mask.sum() < 2:
            continue
        cls_output = output[mask]
        cls_labels = labels[mask]
        with torch.no_grad():
            class_losses[cls.item()] = criterion(cls_output, cls_labels).item()
    return class_losses


device = torch.device("cuda:0")

BATCH_SIZE     = 256

#params for current resolution and column count
IMAGE_SIZE     = 224
PATCH_SIZE     = 1
PATCH_DIM      = 64 # for embedding_dim = 64
CONV_IMG_SIZE  = 14 # 14x14 for the reduced image
NUM_PATCHES_SIDE = CONV_IMG_SIZE // PATCH_SIZE

#other params
N_CHANNELS     = 3
N_CLASSES      = 10
LEVELS         = 2
CONTR_DIM      = 512 #embedding size
DROPOUT        = 0.3
ITERS          = 4
DENOISE_ITER   = -1
LOCAL_CONSENSUS_RADIUS = 0

RUN_NAME = "enc_18"
WORKERS  = 16
EPOCHS = 31
TEMPERATURE = .07
LR = .05

loader          = DALIdataset(run_name=RUN_NAME, downsample=False, batch_size=BATCH_SIZE, num_workers=WORKERS)
img_transformer = IMG_Transformer().to(device)
train_loader    = DALIGenericIterator(loader.train, output_map=["imgs", "subtypes", "tumor_probs"])
val_loader      = DALIGenericIterator(loader.val,   output_map=["imgs", "subtypes", "tumor_probs"])

# model = Agglomerator(
#     name=RUN_NAME,
#     num_patches_side=NUM_PATCHES_SIDE,
#     iters=ITERS,
#     denoise_iter=DENOISE_ITER,
#     n_channels=N_CHANNELS,
#     n_classes=N_CLASSES,
#     levels=LEVELS,
#     patch_dim=PATCH_DIM,
#     contr_dim=CONTR_DIM,
#     conv_image_size=CONV_IMG_SIZE,
#     patch_size=PATCH_SIZE,
#     dropout=DROPOUT,
#     local_consensus_radius=LOCAL_CONSENSUS_RADIUS,
#     toprint=False
# ).to(device)

model = Encoder_18(RUN_NAME).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=5e-4,
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    epochs=EPOCHS,
    steps_per_epoch=len(loader.train_indices) // BATCH_SIZE,
)

SupCon = SupConLoss(temperature=TEMPERATURE)

train_steps = len(loader.train_indices) // BATCH_SIZE
val_steps = len(loader.val_indices) // BATCH_SIZE

for epoch in tqdm(range(loader.epoch, EPOCHS), desc="Epochs"):
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader, total=train_steps, desc=f"  Train {epoch}", leave=False):
        tiles, labels, probs = unpack_dali(batch, device)
        x = img_transformer(tiles)

        top_level, _ = model(x)
        output = rearrange(top_level, '(v b) d -> b v d', v=2) #v=2 for number of images
        loss = SupCon(output, labels)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    train_loss /= train_steps

    model.eval()
    val_loss = 0
    all_class_losses = {}
    with torch.no_grad():
        for batch in tqdm(val_loader, total=val_steps, desc=f"  Val {epoch}", leave=False):
            tiles, labels, probs = unpack_dali(batch, device)
            top_level, _ = model(x)
            output = rearrange(top_level, '(v b) d -> b v d', v=2)
            loss = SupCon(output, labels)
            val_loss += loss.item()
            
            batch_class_losses = per_class_loss(SupCon, output, labels)
            for cls, cls_loss in batch_class_losses.items():
                if cls not in all_class_losses:
                    all_class_losses[cls] = []
                all_class_losses[cls].append(cls_loss)

    val_loss /= val_steps
    class_losses = {cls: sum(losses) / len(losses) 
                for cls, losses in all_class_losses.items()}

    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    loader.write_log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "per_class_loss": class_losses,
        "lr": optimizer.param_groups[0]['lr'],
        # "model": {
        #     "wl":  model.wl.item(),
        #     "wBU": model.wBU.item(),
        #     "wTD": model.wTD.item(),
        #     "wA":  model.wA.item(),
        #     "grad_norm": sum(
        #         p.grad.norm().item() for p in model.parameters() if p.grad is not None
        #     ),
        #     "param_norm": sum(
        #         p.norm().item() for p in model.parameters()
        #     ),
        # },
        "scheduler": {
            "last_lr": scheduler.get_last_lr()[0],
        },
    })
    # model.save(optimizer, scheduler, epoch, train_loss, val_loss)
    model.save()
    train_loader    = DALIGenericIterator(loader.train, output_map=["imgs", "subtypes", "tumor_probs"])
    val_loader      = DALIGenericIterator(loader.val,   output_map=["imgs", "subtypes", "tumor_probs"])
    