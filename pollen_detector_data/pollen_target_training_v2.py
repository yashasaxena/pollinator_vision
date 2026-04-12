#!/usr/bin/env python3
import os
import json
import random

from PIL import ImageOps

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchmetrics.detection import MeanAveragePrecision
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as T

os.makedirs("val_results", exist_ok=True)

# ---------------------------
# Config
# ---------------------------
IMG_ROOT = "images/"
ANN_FILE = "annotations/instances_default.json"
MODEL_OUT = "mobilenet_custom_v9.pth"
NUM_EPOCHS = 100
BATCH_SIZE = 4
LR = 5e-5
VAL_VIS_SCORE_THRESH = 0.3
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Load COCO metadata
# ---------------------------
with open(ANN_FILE, "r") as f:
    coco_data = json.load(f)

categories = coco_data["categories"]
cat_ids = sorted(cat["id"] for cat in categories)

# Remap raw COCO category IDs -> contiguous labels 1..K
cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
label_to_cat_id = {v: k for k, v in cat_id_to_label.items()}
label_to_name = {cat_id_to_label[cat["id"]]: cat["name"] for cat in categories}

num_foreground_classes = len(categories)
num_classes = num_foreground_classes + 1  # +1 for background

print("Categories:")
for label in sorted(label_to_name):
    print(f"  label {label}: {label_to_name[label]}")
print("num_classes =", num_classes)

# ---------------------------
# Dataset
# ---------------------------
class MobileNetDataset(CocoDetection):
    def __init__(self, root, annFile, cat_id_to_label, resize=800):
        super().__init__(root=root, annFile=annFile)
        self.cat_id_to_label = cat_id_to_label
        self.resize = resize

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img = ImageOps.exif_transpose(img)

        boxes = []
        labels = []

        for obj in target:
            x, y, w, h = obj["bbox"]
            if w <= 0 or h <= 0:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[obj["category_id"]])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # ---------------------------
        # Resize full image
        # ---------------------------
        orig_w, orig_h = img.size

        img = T.Resize((self.resize, self.resize))(img)
        img = T.ToTensor()(img)

        # scale boxes
        scale_x = self.resize / orig_w
        scale_y = self.resize / orig_h

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        return img, {
            "boxes": boxes,
            "labels": labels,
        }

full_dataset = MobileNetDataset(
    root=IMG_ROOT,
    annFile=ANN_FILE,
    cat_id_to_label=cat_id_to_label,
)

# ---------------------------
# Train/val split
# ---------------------------
indices = torch.randperm(len(full_dataset)).tolist()
train_size = int(0.8 * len(full_dataset))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

print(f"Total images: {len(full_dataset)}")
print(f"Train images: {len(train_indices)}")
print(f"Val images:   {len(val_indices)}")

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    Subset(full_dataset, train_indices),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=False,
)

val_loader = DataLoader(
    Subset(full_dataset, val_indices),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=False,
)

# ---------------------------
# Model
# ---------------------------
# model = ssdlite320_mobilenet_v3_large(
#     weights_backbone="DEFAULT",
#     num_classes=num_classes
# )
model = fasterrcnn_mobilenet_v3_large_fpn(
    weights=None,
    num_classes=num_classes
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ---------------------------
# Training
# ---------------------------
print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
        num_batches += 1

    avg_loss = epoch_loss / max(num_batches, 1)
    print(f"Epoch {epoch + 1:03d}/{NUM_EPOCHS} - Avg Loss: {avg_loss:.4f}")

# ---------------------------
# Evaluation
# ---------------------------
print("\nCalculating validation mAP...")
model.eval()
metric = MeanAveragePrecision(class_metrics=True)

saved_img_idx = 0

with torch.no_grad():
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        preds = []
        gts = []

        for output, target in zip(outputs, targets):
            preds.append({
                "boxes": output["boxes"].cpu(),
                "scores": output["scores"].cpu(),
                "labels": output["labels"].cpu(),
            })
            gts.append({
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
            })

        metric.update(preds, gts)

        # Save visualizations
        for img, output in zip(images, outputs):
            keep = output["scores"] > VAL_VIS_SCORE_THRESH
            boxes = output["boxes"][keep].detach().cpu()
            scores = output["scores"][keep].detach().cpu()
            labels_tensor = output["labels"][keep].detach().cpu()

            text_labels = [
                f"{label_to_name.get(lbl.item(), f'class_{lbl.item()}')}: {score.item():.2f}"
                for lbl, score in zip(labels_tensor, scores)
            ]

            img_uint8 = (img.detach().cpu() * 255).to(torch.uint8)

            if len(boxes) > 0:
                result_img = draw_bounding_boxes(
                    img_uint8,
                    boxes=boxes,
                    labels=text_labels,
                    colors="red",
                    width=2,
                )
            else:
                result_img = img_uint8

            result_pil = F.to_pil_image(result_img)
            result_pil.save(f"val_results/val_img_{saved_img_idx}.jpg")
            saved_img_idx += 1

results = metric.compute()

print("\n--- Validation Metrics ---")
for k, v in results.items():
    if torch.is_tensor(v) and v.numel() == 1:
        print(f"{k}: {v.item():.4f}")
    elif torch.is_tensor(v):
        print(f"{k}: {v}")

# Per-class AP
map_per_class = results.get("map_per_class")
classes = results.get("classes")

if map_per_class is not None and classes is not None:
    print("\n--- Per-class AP ---")
    for class_label, ap in zip(classes.tolist(), map_per_class.tolist()):
        class_name = label_to_name.get(class_label, f"class_{class_label}")
        print(f"{class_name} (label {class_label}): {ap:.2%}")

# ---------------------------
# Save
# ---------------------------
torch.save(model.state_dict(), MODEL_OUT)
print(f"\nModel saved as {MODEL_OUT}")