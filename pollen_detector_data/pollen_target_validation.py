#!/usr/bin/env python3
import os
import json
import torch
import torchvision

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchmetrics.detection import MeanAveragePrecision

import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes

# ---------------------------
# Config
# ---------------------------
IMG_ROOT = "images/"
ANN_FILE = "annotations/instances_default.json"
MODEL_PATH = "mobilenet_custom_v4.pth"

BATCH_SIZE = 4
SAVE_IMAGES = True
CONF_THRESH = 0.3

os.makedirs("val_results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Load annotations + mapping
# ---------------------------
with open(ANN_FILE, "r") as f:
    coco_data = json.load(f)

categories = coco_data["categories"]
cat_ids = sorted(cat["id"] for cat in categories)

cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
label_to_name = {cat_id_to_label[cat["id"]]: cat["name"] for cat in categories}

num_classes = len(categories) + 1

print("Classes:")
for k, v in label_to_name.items():
    print(k, "->", v)

# ---------------------------
# Dataset
# ---------------------------
class MobileNetDataset(CocoDetection):
    def __init__(self, root, annFile, cat_id_to_label):
        super().__init__(root=root, annFile=annFile)
        self.cat_id_to_label = cat_id_to_label
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

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

        return self.to_tensor(img), {"boxes": boxes, "labels": labels}

# ---------------------------
# Data split (same as training!)
# ---------------------------
torch.manual_seed(42)

full_dataset = MobileNetDataset(
    root=IMG_ROOT,
    annFile=ANN_FILE,
    cat_id_to_label=cat_id_to_label,
)

indices = torch.randperm(len(full_dataset)).tolist()
train_size = int(0.8 * len(full_dataset))

val_indices = indices[train_size:]

val_loader = DataLoader(
    Subset(full_dataset, val_indices),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x)),
)

print(f"Validation images: {len(val_indices)}")

# ---------------------------
# Load model
# ---------------------------
model = ssdlite320_mobilenet_v3_large(
    weights_backbone="DEFAULT",
    num_classes=num_classes
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Model loaded.")

# ---------------------------
# Evaluation
# ---------------------------
metric = MeanAveragePrecision(class_metrics=True)

img_id = 0

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

        # Optional visualization
        if SAVE_IMAGES:
            for img, output in zip(images, outputs):
                keep = output["scores"] > CONF_THRESH

                boxes = output["boxes"][keep].cpu()
                scores = output["scores"][keep].cpu()
                labels = output["labels"][keep].cpu()

                text_labels = [
                    f"{label_to_name[l.item()]} {s.item():.2f}"
                    for l, s in zip(labels, scores)
                ]

                img_uint8 = (img.cpu() * 255).to(torch.uint8)

                if len(boxes) > 0:
                    drawn = draw_bounding_boxes(
                        img_uint8,
                        boxes,
                        labels=text_labels,
                        colors="pink",
                        width=10,
                    )
                else:
                    drawn = img_uint8

                F.to_pil_image(drawn).save(f"val_results/val_{img_id}.jpg")
                img_id += 1

# ---------------------------
# Results
# ---------------------------
results = metric.compute()

print("\n--- Metrics ---")
for k, v in results.items():
    if torch.is_tensor(v) and v.numel() == 1:
        print(f"{k}: {v.item():.4f}")

print("\n--- Per-class AP ---")
for cls, ap in zip(results["classes"], results["map_per_class"]):
    name = label_to_name.get(cls.item(), str(cls.item()))
    print(f"{name}: {ap.item():.2%}")