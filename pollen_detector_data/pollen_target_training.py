import os
import torch
import torchvision
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchmetrics.detection import MeanAveragePrecision
import os
import json

import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import PIL.Image as Image


os.makedirs('val_results', exist_ok=True)

# 1. Custom Dataset Loader for COCO
class MobileNetDataset(CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        # MobileNet SSDLite expects: boxes [x1, y1, x2, y2] and labels
        boxes = []
        labels = []
        for obj in target:
            x, y, w, h = obj['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(obj['category_id'])

        target_res = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }
        # Transform image to Tensor
        img = torchvision.transforms.ToTensor()(img)
        return img, target_res

# 2. Setup Data & Split (80/20)
full_dataset = MobileNetDataset(root='images/', annFile='annotations/instances_default.json')
# Load the category names from the CVAT JSON file
with open('annotations/instances_default.json', 'r') as f:
    coco_data = json.load(f)
category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
indices = torch.randperm(len(full_dataset)).tolist()
train_size = int(0.8 * len(full_dataset))

train_loader = DataLoader(
    Subset(full_dataset, indices[:train_size]), 
    batch_size=4, 
    shuffle=True, 
    collate_fn=lambda x: tuple(zip(*x)),
    drop_last=True  # <--- THIS PREVENTS THE CRASH
)

val_loader = DataLoader(
    Subset(full_dataset, indices[train_size:]), 
    batch_size=4, 
    shuffle=False, 
    collate_fn=lambda x: tuple(zip(*x)),
    drop_last=False # Validation doesn't usually use BatchNorm in 'train' mode, so this is fine
)
# 3. Initialize MobileNetV3 SSDLite FROM SCRATCH
# num_classes = your labels + 1 (for background)
# weights=None ensures no pre-training is used
model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=4) 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# # 4. Training Loop
# print("Starting Training...")
# for epoch in range(200): # Adjust epochs as needed
#     model.train()
#     for images, targets in train_loader:
#         images = [img.to(device) for img in images]
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
        
#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()
#     print(f"Epoch {epoch} Loss: {losses.item():.4f}")

model.load_state_dict(torch.load("mobilenet_custom_v3.pth", map_location=device))
model.to(device)

# 5. Accuracy Report by Label Type
print("\nCalculating Per-Label Accuracy (mAP)...")
model.eval()
metric = MeanAveragePrecision(class_metrics=True)

with torch.no_grad():
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        # Format for torchmetrics
        res = [{"boxes": o["boxes"].cpu(), "scores": o["scores"].cpu(), "labels": o["labels"].cpu()} for o in outputs]
        gt = [{"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()} for t in targets]
        metric.update(res, gt)

        # Inside your for images, targets in val_loader:
        for i, (img, output) in enumerate(zip(images, outputs)):
            # 1. Select only high-confidence predictions
            # keep = output['scores'] > 0.3  # Only show boxes with >50% confidence
            keep = torch.ones_like(output['scores'], dtype=torch.bool) 
            boxes = output['boxes'][keep]
            labels = [f"Label {l.item()}: {s.item():.2f}" for l, s in zip(output['labels'][keep], output['scores'][keep])]

            # 2. Convert image tensor back to uint8 format (0-255) for drawing
            img_uint8 = (img * 255).to(torch.uint8)

            # 3. Draw bounding boxes on the image
            result_img = draw_bounding_boxes(img_uint8, boxes=boxes, labels=labels, colors="red", width=3)

            # 4. Save to your local folder
            result_pil = F.to_pil_image(result_img)
            result_pil.save(f"val_results/val_img_{i}.jpg")

results = metric.compute()
print("\n--- Accuracy Report (mAP) ---")
# The index 'i' in map_per_class usually corresponds to the sorted category IDs
sorted_ids = sorted(category_map.keys())

for i, ap in enumerate(results['map_per_class']):
    cat_id = sorted_ids[i]
    cat_name = category_map[cat_id]
    print(f"Label '{cat_name}' (ID {cat_id}) Accuracy: {ap.item():.2%}")

# 6. Save for Raspberry Pi
torch.save(model.state_dict(), "mobilenet_custom_v3.pth")
print("\nModel saved as mobilenet_custom_v3.pth")