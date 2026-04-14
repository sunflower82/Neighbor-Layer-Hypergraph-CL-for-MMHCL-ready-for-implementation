"""
reextract_image_features.py
============================
Re-runs CLIP image feature extraction with corrected HTTPS URLs.
The original preprocess script used http:// which is now blocked.
This script converts to https:// and re-downloads all images.
"""

import json, os, numpy as np, torch, requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from preprocess_helpers import iter_gzip_jsonlines, select_image_url, clip_image_embeddings

BASE     = os.path.join(os.path.dirname(__file__), "data", "Baby")
META_GZ  = os.path.join(BASE, "meta_Baby.json.gz")
ITEM2ID  = os.path.join(BASE, "item2id.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'N/A'}")

# Load item ID mapping from prior preprocessing
with open(ITEM2ID) as f:
    item2id: dict[str, int] = json.load(f)
NUM_ITEMS = len(item2id)
print(f"Items: {NUM_ITEMS:,}")

# Load CLIP model
print("Loading CLIP ViT-B/32...")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("CLIP model loaded.")

# Read metadata — collect image URLs with HTTPS fix
print("Reading metadata and fixing image URLs...")
item_urls: dict[int, str] = {}

for d in iter_gzip_jsonlines(META_GZ):
    asin = d.get("asin", "")
    if asin not in item2id:
        continue
    url = select_image_url(d, force_https=True)
    if url:
        item_urls[item2id[asin]] = url

print(f"URLs collected: {len(item_urls):,} / {NUM_ITEMS:,}")

# Extract image features
image_features = np.zeros((NUM_ITEMS, 512), dtype=np.float32)
ok = 0
fail = 0

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

with torch.no_grad():
    for iid in tqdm(range(NUM_ITEMS), desc="Image"):
        url = item_urls.get(iid, "")
        if not url:
            fail += 1
            continue
        try:
            resp = session.get(url, timeout=5)
            resp.raise_for_status()
            img  = Image.open(BytesIO(resp.content)).convert("RGB")
            inp  = processor(images=img, return_tensors="pt").to(DEVICE)

            emb = clip_image_embeddings(model, inp)
            image_features[iid] = emb.cpu().numpy().flatten()
            ok += 1
        except Exception:
            fail += 1

out_path = os.path.join(BASE, "image_feat.npy")
np.save(out_path, image_features)

print(f"\n{'='*60}")
print(f"Image feature extraction complete")
print(f"  Saved : {out_path}")
print(f"  Shape : {image_features.shape}")
print(f"  OK    : {ok:,} / {NUM_ITEMS:,}")
print(f"  Failed: {fail:,}  (zero-vectors used)")
non_zero = np.count_nonzero(image_features.sum(axis=1))
print(f"  Non-zero rows: {non_zero:,}")
