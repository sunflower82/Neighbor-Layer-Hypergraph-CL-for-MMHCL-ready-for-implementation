"""
preprocess_amazon_baby.py
=========================
Preprocessing script for the Amazon Baby dataset for use with MMHCL.

Evaluation protocol (matching BM3, WWW '23):
  - **Per-user 8:1:1 random split**: for each user, randomly shuffle their
    interactions and split into ~80% train, ~10% val, ~10% test.
    Every user is guaranteed at least 1 item in each split (train/val/test).
    This ensures all 19,445 users are evaluated (no user selection bias).
  - CLIP ViT-B/32 for 512-dimensional image and text features

Output files produced (matching MMHCL load_data.py expectations):
  data/Baby/5-core/train.json   -- {uid_str: [iid, ...], ...}
  data/Baby/5-core/val.json
  data/Baby/5-core/test.json
  data/Baby/image_feat.npy      -- (n_items, 512) float32
  data/Baby/text_feat.npy       -- (n_items, 512) float32

Run from the project root:
  python preprocess_amazon_baby.py
"""

from collections import defaultdict
import gzip
from io import BytesIO
import json
import math
import os
import random

import numpy as np
from PIL import Image
import requests
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ===========================================================================
#  Paths
# ===========================================================================
BASE = os.path.join(os.path.dirname(__file__), "data", "Baby")
REVIEW_GZ = os.path.join(BASE, "reviews_Baby_5.json.gz")
META_GZ = os.path.join(BASE, "meta_Baby.json.gz")
CORE_DIR = os.path.join(BASE, "5-core")
os.makedirs(CORE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ===========================================================================
#  PART 1 — ID mapping and per-user 8:1:1 random split (BM3 protocol)
# ===========================================================================
print("\n" + "=" * 60)
print("PART 1: Reading reviews and building ID mappings")
print("=" * 60)

interactions = []
with gzip.open(REVIEW_GZ, "rb") as f:
    for line in f:
        try:
            d = json.loads(line)
            interactions.append(
                (
                    d["reviewerID"],
                    d["asin"],
                )
            )
        except Exception:
            continue

print(f"Total interactions read : {len(interactions):,}")

# Build continuous integer ID mappings (sorted for reproducibility)
all_users = sorted({r[0] for r in interactions})
all_items = sorted({r[1] for r in interactions})
user2id = {u: i for i, u in enumerate(all_users)}
item2id = {v: i for i, v in enumerate(all_items)}
NUM_USERS = len(user2id)
NUM_ITEMS = len(item2id)
print(f"Unique users           : {NUM_USERS:,}")
print(f"Unique items           : {NUM_ITEMS:,}")

# Group interactions by user
user_items: dict[int, list[int]] = defaultdict(list)
for reviewer, asin in interactions:
    uid = user2id[reviewer]
    iid = item2id[asin]
    user_items[uid].append(iid)

# Per-user 8:1:1 random split (matching BM3, WWW '23)
# For each user with n items:
#   n_test = max(1, floor(n * 0.1))
#   n_val  = max(1, floor(n * 0.1))
#   n_train = n - n_test - n_val
# Guarantees every user has >= 1 item in train, val, and test.
random.seed(42)

train_dict: dict[str, list[int]] = {}
val_dict: dict[str, list[int]] = {}
test_dict: dict[str, list[int]] = {}

for uid in range(NUM_USERS):
    items = user_items[uid][:]
    random.shuffle(items)
    n = len(items)

    n_test = max(1, math.floor(n * 0.1))
    n_val = max(1, math.floor(n * 0.1))

    test_items = items[:n_test]
    val_items = items[n_test : n_test + n_val]
    train_items = items[n_test + n_val :]

    train_dict[str(uid)] = train_items
    val_dict[str(uid)] = val_items
    test_dict[str(uid)] = test_items

n_train = sum(len(v) for v in train_dict.values())
n_val = sum(len(v) for v in val_dict.values())
n_test = sum(len(v) for v in test_dict.values())
total = n_train + n_val + n_test
print("\nSplit summary:")
print(f"  Train : {n_train:>7,}  ({100 * n_train / total:.1f}%)")
print(f"  Val   : {n_val:>7,}  ({100 * n_val / total:.1f}%)")
print(f"  Test  : {n_test:>7,}  ({100 * n_test / total:.1f}%)")

# Save JSON files
for name, data in [
    ("train.json", train_dict),
    ("val.json", val_dict),
    ("test.json", test_dict),
]:
    path = os.path.join(CORE_DIR, name)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Saved: {path}")

# Save ID mappings for reference
with open(os.path.join(BASE, "user2id.json"), "w") as f:
    json.dump(user2id, f)
with open(os.path.join(BASE, "item2id.json"), "w") as f:
    json.dump(item2id, f)
print("\nID mappings saved to data/Baby/user2id.json and item2id.json")

# ===========================================================================
#  PART 2 — Read metadata: build text and image-URL lookup per item
# ===========================================================================
print("\n" + "=" * 60)
print("PART 2: Reading metadata")
print("=" * 60)

item_meta: dict[int, dict] = {}
meta_found = 0

with gzip.open(META_GZ, "rb") as f:
    for line in f:
        try:
            # Amazon metadata files may use Python-eval format, not strict JSON
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                d = eval(line)

            asin = d.get("asin", "")
            if asin not in item2id:
                continue

            iid = item2id[asin]
            title = d.get("title", "") or ""
            desc = d.get("description", "") or ""
            if isinstance(desc, list):
                desc = " ".join(str(x) for x in desc)

            # Prefer high-resolution images, fall back to imUrl
            img_url = ""
            if d.get("imageURLHighRes"):
                img_url = d["imageURLHighRes"][0] if d["imageURLHighRes"] else ""
            if not img_url:
                img_url = d.get("imUrl", "") or ""

            item_meta[iid] = {
                "text": f"{title} {desc}".strip(),
                "image_url": img_url,
            }
            meta_found += 1
        except Exception:
            continue

print(f"Metadata matched for {meta_found:,} / {NUM_ITEMS:,} items")
print(f"  Items without metadata (features will be zero): {NUM_ITEMS - meta_found:,}")

# ===========================================================================
#  PART 3 — CLIP ViT-B/32 feature extraction
# ===========================================================================
print("\n" + "=" * 60)
print("PART 3: Loading CLIP ViT-B/32 model")
print("=" * 60)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("CLIP model loaded.")

# Allocate output matrices
image_features = np.zeros((NUM_ITEMS, 512), dtype=np.float32)
text_features = np.zeros((NUM_ITEMS, 512), dtype=np.float32)

# ---- 3a. Batch text encoding ----
print("\nExtracting text features (batch_size=128)...")
TEXT_BATCH = 128
item_ids_list = list(range(NUM_ITEMS))


def _get_text_embs(model, inputs):
    """Extract (B, 512) text embeddings — compatible with transformers v4 and v5."""
    result = model.get_text_features(**inputs)
    if isinstance(result, torch.Tensor):
        return result
    # transformers v5+ may return a ModelOutput dataclass
    if hasattr(result, "pooler_output"):
        return model.text_projection(result.pooler_output)
    return model.text_projection(result[1])  # fallback: index 1 = pooler_output


with torch.no_grad():
    for start in tqdm(range(0, NUM_ITEMS, TEXT_BATCH), desc="Text"):
        batch_ids = item_ids_list[start : start + TEXT_BATCH]
        texts = []
        for iid in batch_ids:
            raw = item_meta.get(iid, {}).get("text", "").strip()
            texts.append(raw if raw else "unknown product")

        inputs = processor(
            text=texts,
            return_tensors="pt",
            truncation=True,
            max_length=77,
            padding=True,
        ).to(DEVICE)
        embs = _get_text_embs(model, inputs)  # (B, 512)
        text_features[start : start + TEXT_BATCH] = embs.cpu().numpy()

np.save(os.path.join(BASE, "text_feat.npy"), text_features)
print(f"  Saved: data/Baby/text_feat.npy  shape={text_features.shape}")

# ---- 3b. Image encoding (one at a time — handles URL failures gracefully) ----
print("\nExtracting image features (1-by-1, 3s URL timeout)...")
img_ok = 0
img_fail = 0

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

with torch.no_grad():
    for iid in tqdm(range(NUM_ITEMS), desc="Image"):
        url = item_meta.get(iid, {}).get("image_url", "")
        if not url:
            img_fail += 1
            continue
        try:
            resp = session.get(url, timeout=3)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            inp = processor(images=img, return_tensors="pt").to(DEVICE)
            raw_img = model.get_image_features(**inp)
            emb = (
                raw_img
                if isinstance(raw_img, torch.Tensor)
                else (
                    model.visual_projection(raw_img.pooler_output)
                    if hasattr(raw_img, "pooler_output")
                    else model.visual_projection(raw_img[1])
                )
            )
            image_features[iid] = emb.cpu().numpy().flatten()
            img_ok += 1
        except Exception:
            img_fail += 1

np.save(os.path.join(BASE, "image_feat.npy"), image_features)
print(f"  Saved: data/Baby/image_feat.npy  shape={image_features.shape}")
print(f"  Images downloaded OK : {img_ok:,}")
print(f"  Images failed/missing: {img_fail:,}  (zero-vectors used)")

# ===========================================================================
#  Summary
# ===========================================================================
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print(f"  data/Baby/5-core/train.json  — {n_train:,} interactions")
print(f"  data/Baby/5-core/val.json    — {n_val:,} interactions")
print(f"  data/Baby/5-core/test.json   — {n_test:,} interactions")
print(f"  data/Baby/text_feat.npy      — shape {text_features.shape}")
print(f"  data/Baby/image_feat.npy     — shape {image_features.shape}")
print("\nNext: run training with --dataset Baby")
