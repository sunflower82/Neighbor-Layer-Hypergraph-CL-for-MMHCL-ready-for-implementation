"""
verify_and_fix_text_features.py
================================
Checks the existing text_feat.npy for correctness and re-extracts if needed.
Also verifies all 5 output files required by MMHCL.
"""
import os, json, numpy as np, torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from preprocess_helpers import iter_gzip_jsonlines, clip_text_embeddings

BASE     = os.path.join(os.path.dirname(__file__), "data", "Baby")
CORE_DIR = os.path.join(BASE, "5-core")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# 1. Quick sanity check on existing text_feat.npy
# ------------------------------------------------------------------
text_path = os.path.join(BASE, "text_feat.npy")
text_feat  = np.load(text_path)
n_nonzero  = np.count_nonzero(text_feat.sum(axis=1))
print(f"text_feat.npy  shape={text_feat.shape}  non-zero rows={n_nonzero:,}")

if n_nonzero < text_feat.shape[0] * 0.5:
    print("WARNING: more than 50% of text features are zero — re-extracting...")
    REEXTRACT_TEXT = True
else:
    print("Text features look OK — skipping re-extraction.")
    REEXTRACT_TEXT = False

# ------------------------------------------------------------------
# 2. Re-extract text features if needed
# ------------------------------------------------------------------
if REEXTRACT_TEXT:
    META_GZ  = os.path.join(BASE, "meta_Baby.json.gz")
    ITEM2ID  = os.path.join(BASE, "item2id.json")
    with open(ITEM2ID) as f:
        item2id = json.load(f)
    NUM_ITEMS = len(item2id)

    print("Loading CLIP ViT-B/32 for text re-extraction...")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    item_texts: dict[int, str] = {}
    for d in iter_gzip_jsonlines(META_GZ):
        asin = d.get("asin", "")
        if asin not in item2id:
            continue
        iid   = item2id[asin]
        title = d.get("title", "") or ""
        desc  = d.get("description", "") or ""
        if isinstance(desc, list):
            desc = " ".join(str(x) for x in desc)
        item_texts[iid] = f"{title} {desc}".strip() or "unknown product"

    text_features = np.zeros((NUM_ITEMS, 512), dtype=np.float32)
    TEXT_BATCH = 128
    with torch.no_grad():
        for start in tqdm(range(0, NUM_ITEMS, TEXT_BATCH), desc="Text"):
            batch_ids = list(range(start, min(start + TEXT_BATCH, NUM_ITEMS)))
            texts     = [item_texts.get(i, "unknown product") for i in batch_ids]
            inputs    = processor(
                text=texts, return_tensors="pt",
                truncation=True, max_length=77, padding=True
            ).to(DEVICE)
            emb = clip_text_embeddings(model, inputs)
            text_features[start: start + len(batch_ids)] = emb.cpu().numpy()

    np.save(text_path, text_features)
    print(f"Re-saved text_feat.npy  shape={text_features.shape}")
    n_nonzero = np.count_nonzero(text_features.sum(axis=1))
    print(f"Non-zero rows: {n_nonzero:,}")

# ------------------------------------------------------------------
# 3. Final verification of all required files
# ------------------------------------------------------------------
print("\n" + "="*60)
print("FINAL VERIFICATION")
print("="*60)

files_to_check = {
    "5-core/train.json" : CORE_DIR + "/train.json",
    "5-core/val.json"   : CORE_DIR + "/val.json",
    "5-core/test.json"  : CORE_DIR + "/test.json",
    "image_feat.npy"    : os.path.join(BASE, "image_feat.npy"),
    "text_feat.npy"     : os.path.join(BASE, "text_feat.npy"),
}

all_ok = True
for name, path in files_to_check.items():
    if not os.path.exists(path):
        print(f"  [MISSING] {name}")
        all_ok = False
        continue
    size_mb = os.path.getsize(path) / 1e6
    if name.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        n_users = len(data)
        n_ints  = sum(len(v) for v in data.values())
        print(f"  [OK]  {name:<25}  {size_mb:6.1f} MB  |  {n_users:,} users  {n_ints:,} interactions")
    else:
        arr = np.load(path)
        n_nz = np.count_nonzero(arr.sum(axis=1))
        print(f"  [OK]  {name:<25}  {size_mb:6.1f} MB  |  shape={arr.shape}  non-zero={n_nz:,}")

print()
if all_ok:
    print("All 5 files present and valid.")
    print("Ready to train: python codes/main.py --dataset Baby ...")
else:
    print("Some files are missing! Check the errors above.")
