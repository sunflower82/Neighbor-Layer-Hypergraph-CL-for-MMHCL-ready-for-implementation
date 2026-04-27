"""
remap_clothing_orphans.py — P3: Lược bỏ item mồ côi (orphan) và remap ID
========================================================================

Mục tiêu
--------
Trong dataset Amazon Clothing 5-core đang dùng cho MMHCL+ Rev5.2, có 7,509/23,033
(32.6%) item KHÔNG xuất hiện ở bất kỳ split nào (train/val/test). Những item
này vẫn chiếm slot trong `n_items` → `Item_mat` và hypergraph có chiều 23,033²
thay vì 15,524². Script này:

  1. Quét union các item trong train ∪ val ∪ test
  2. Tạo bảng map old_id → new_id (giữ thứ tự tăng dần để dễ debug)
  3. Viết lại train.json, val.json, test.json với new_id
  4. Lọc image_feat.npy / text_feat.npy theo old_id (giữ đúng thứ tự)
  5. (Tuỳ chọn) Lược user không có train interaction nào
  6. Xoá cache .pth/.npz cũ (vì kích thước đã đổi)
  7. Backup tất cả file gốc vào <data_path>/_orig_backup/

Cách dùng
---------
    # Dry-run trước (không ghi gì, chỉ in thống kê)
    python remap_clothing_orphans.py --data-path data/Clothing --dry-run

    # Thực thi (mặc định backup originals tự động)
    python remap_clothing_orphans.py --data-path data/Clothing

    # Chỉ remap item, không động đến user (mặc định)
    python remap_clothing_orphans.py --data-path data/Clothing --no-remap-users

    # Remap cả user (chỉ làm nếu có user gap)
    python remap_clothing_orphans.py --data-path data/Clothing --remap-users

Sau khi chạy xong
-----------------
    1. Kiểm tra bằng:  python verify_remap.py --data-path data/Clothing
    2. Chạy huấn luyện trên 1 seed để so sánh NDCG@20 với baseline 0.0378
    3. Nếu sai lệch < 0.5%, mở rộng cho cả 3 seed

Lưu ý
-----
- Script ghi log chi tiết + lưu metadata.json để có thể "rollback"
- Tất cả file gốc backup vào _orig_backup/<timestamp>/
- KHÔNG xóa data gốc; chỉ ghi đè khi đã backup
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# ===========================================================================
#  CLI parser
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lược orphan items và remap ID cho Amazon Clothing 5-core",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-path",
        type=str,
        default="data/Clothing",
        help="Thư mục dataset (chứa 5-core/ và *.npy)",
    )
    p.add_argument(
        "--core",
        type=int,
        default=5,
        help="K-core suffix (5 → đọc 5-core/train.json)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ in thống kê, không ghi/xóa file nào",
    )
    p.add_argument(
        "--remap-users",
        dest="remap_users",
        action="store_true",
        help="Remap cả user ID (chỉ cần nếu có gap trong user IDs)",
    )
    p.add_argument(
        "--no-remap-users",
        dest="remap_users",
        action="store_false",
        help="Không remap user (mặc định)",
    )
    p.set_defaults(remap_users=False)
    p.add_argument(
        "--features",
        nargs="*",
        default=["image_feat.npy", "text_feat.npy"],
        help="Tên file feature cần lọc theo item old_id",
    )
    p.add_argument(
        "--purge-cache",
        action="store_true",
        default=True,
        help="Xoá *.pth, *.npz cache cũ trong <core>-core/ (kích thước đã đổi)",
    )
    p.add_argument(
        "--no-purge-cache",
        dest="purge_cache",
        action="store_false",
    )
    return p.parse_args()


# ===========================================================================
#  Helpers
# ===========================================================================
def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _read_split(path: Path) -> dict[int, list[int]]:
    """Đọc JSON split, ép key thành int và lọc value là list of int."""
    raw = json.loads(path.read_text())
    out: dict[int, list[int]] = {}
    for k, v in raw.items():
        uid = int(k)
        if not isinstance(v, list):
            continue
        # Ép từng item về int — JSON đôi khi có số nguyên kiểu str
        out[uid] = [int(i) for i in v]
    return out


def _scan_universe(
    train: dict[int, list[int]],
    val: dict[int, list[int]],
    test: dict[int, list[int]],
) -> tuple[set[int], set[int], int, int]:
    """
    Returns: (used_users, used_items, n_users_old, n_items_old)
    used_users = các user có ≥1 interaction trong train (để loại user orphan)
    used_items = union các item xuất hiện trong train ∪ val ∪ test
    """
    used_users: set[int] = set()
    used_items: set[int] = set()
    n_users_old = 0
    n_items_old = 0

    for uid, items in train.items():
        if items:
            used_users.add(uid)
            used_items.update(items)
            n_users_old = max(n_users_old, uid)
            n_items_old = max(n_items_old, max(items))

    for uid, items in val.items():
        if items:
            n_users_old = max(n_users_old, uid)
            n_items_old = max(n_items_old, max(items))
            used_items.update(items)

    for uid, items in test.items():
        if items:
            n_users_old = max(n_users_old, uid)
            n_items_old = max(n_items_old, max(items))
            used_items.update(items)

    return used_users, used_items, n_users_old + 1, n_items_old + 1


def _build_id_map(used_ids: set[int]) -> dict[int, int]:
    """Map old_id → new_id, sorted ascending để giữ thứ tự gốc."""
    return {old: new for new, old in enumerate(sorted(used_ids))}


def _remap_split(
    split: dict[int, list[int]],
    user_map: dict[int, int] | None,
    item_map: dict[int, int],
) -> tuple[dict[str, list[int]], int, int]:
    """
    Remap một split. Nếu user_map=None thì giữ nguyên user IDs.
    Trả về (new_split_dict_str_keys, n_interactions, n_dropped_pairs).
    """
    out: dict[str, list[int]] = {}
    n_pairs = 0
    n_dropped = 0
    for uid, items in split.items():
        new_uid = user_map[uid] if user_map is not None else uid
        new_items: list[int] = []
        for i in items:
            if i in item_map:
                new_items.append(item_map[i])
            else:
                n_dropped += 1
        if new_items:
            out[str(new_uid)] = new_items
            n_pairs += len(new_items)
    return out, n_pairs, n_dropped


def _filter_features(
    feat_path: Path,
    used_items_sorted: list[int],
    out_path: Path,
    dry_run: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Đọc .npy, lọc theo used_items_sorted (giữ đúng thứ tự new_id), ghi đè.
    Trả về (old_shape, new_shape).
    """
    arr = np.load(feat_path, mmap_mode="r")
    old_shape = arr.shape
    # used_items_sorted đã sorted ascending → new_id i tương ứng old_id used_items_sorted[i]
    indices = np.asarray(used_items_sorted, dtype=np.int64)
    if indices.max() >= arr.shape[0]:
        raise ValueError(
            f"Feature {feat_path.name} có shape {arr.shape} nhưng max old_id = "
            f"{indices.max()} (out of bounds). Feature file có khớp với JSON không?"
        )
    new_arr = np.ascontiguousarray(arr[indices])
    new_shape = new_arr.shape
    if not dry_run:
        np.save(out_path, new_arr)
    return old_shape, new_shape


# ===========================================================================
#  Main
# ===========================================================================
def main() -> int:
    args = parse_args()

    data_root = Path(args.data_path).resolve()
    core_dir = data_root / f"{args.core}-core"

    if not core_dir.exists():
        _log(f"KHÔNG tìm thấy {core_dir}")
        return 1

    train_path = core_dir / "train.json"
    val_path = core_dir / "val.json"
    test_path = core_dir / "test.json"

    for p in (train_path, val_path, test_path):
        if not p.exists():
            _log(f"Thiếu file {p}")
            return 1

    _log(f"Đọc dataset từ {core_dir}")
    train = _read_split(train_path)
    val = _read_split(val_path)
    test = _read_split(test_path)

    used_users, used_items, n_users_old, n_items_old = _scan_universe(
        train, val, test
    )

    n_users_used = len(used_users)
    n_items_used = len(used_items)
    n_orphan_items = n_items_old - n_items_used
    n_orphan_users = n_users_old - n_users_used

    _log("─" * 64)
    _log("THỐNG KÊ TRƯỚC KHI REMAP")
    _log("─" * 64)
    _log(f"  n_users (old, dense range)   = {n_users_old:,}")
    _log(f"  n_items (old, dense range)   = {n_items_old:,}")
    _log(f"  users có ≥1 train interaction = {n_users_used:,}")
    _log(f"  items xuất hiện trong train ∪ val ∪ test = {n_items_used:,}")
    _log(f"  → orphan items (không split nào dùng)   = {n_orphan_items:,}")
    _log(
        f"  → orphan users (không có train interaction) = {n_orphan_users:,}"
    )
    _log(
        f"  Tỷ lệ giảm n_items: {n_items_used/n_items_old*100:.1f}% "
        f"(tiết kiệm ~{(1 - (n_items_used/n_items_old)**2)*100:.0f}% dung lượng Item_mat)"
    )

    if n_orphan_items == 0 and (not args.remap_users or n_orphan_users == 0):
        _log("Không có orphan nào — không cần remap.")
        return 0

    # ─── Build maps ────────────────────────────────────────────────────────
    item_map = _build_id_map(used_items)
    user_map: dict[int, int] | None = None
    if args.remap_users:
        user_map = _build_id_map(used_users)
        _log(
            f"User remap được bật: {n_users_old:,} → {len(user_map):,} "
            f"(loại {n_orphan_users:,})"
        )
    else:
        if n_orphan_users > 0:
            _log(
                f"Cảnh báo: có {n_orphan_users:,} user gap nhưng --no-remap-users; "
                "model sẽ giữ slot embedding rỗng cho các user đó."
            )

    # ─── Remap splits ──────────────────────────────────────────────────────
    new_train, n_train_pairs, dropped_train = _remap_split(train, user_map, item_map)
    new_val, n_val_pairs, dropped_val = _remap_split(val, user_map, item_map)
    new_test, n_test_pairs, dropped_test = _remap_split(test, user_map, item_map)

    _log("─" * 64)
    _log("THỐNG KÊ SAU KHI REMAP")
    _log("─" * 64)
    _log(
        f"  n_train_pairs: {n_train_pairs:,} "
        f"(dropped {dropped_train} cặp do item bị xoá)"
    )
    _log(
        f"  n_val_pairs:   {n_val_pairs:,} "
        f"(dropped {dropped_val} cặp do item bị xoá)"
    )
    _log(
        f"  n_test_pairs:  {n_test_pairs:,} "
        f"(dropped {dropped_test} cặp do item bị xoá)"
    )
    if dropped_train + dropped_val + dropped_test > 0:
        _log(
            "  Lưu ý: drop chỉ xảy ra khi --remap-users bật và user bị loại; "
            "nếu chỉ remap item, drop phải = 0."
        )

    # ─── Dry-run thoát sớm ────────────────────────────────────────────────
    if args.dry_run:
        _log("DRY-RUN: không ghi gì. Chạy lại không có --dry-run để áp dụng.")
        return 0

    # ─── Backup ────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = data_root / "_orig_backup" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_core = backup_dir / f"{args.core}-core"
    backup_core.mkdir(parents=True, exist_ok=True)

    _log(f"Backup originals vào {backup_dir}")
    for p in (train_path, val_path, test_path):
        shutil.copy2(p, backup_core / p.name)
    for fname in args.features:
        fp = data_root / fname
        if fp.exists():
            shutil.copy2(fp, backup_dir / fname)

    # ─── Ghi splits mới ────────────────────────────────────────────────────
    _log("Ghi splits mới (giữ nguyên đường dẫn gốc)")
    train_path.write_text(json.dumps(new_train))
    val_path.write_text(json.dumps(new_val))
    test_path.write_text(json.dumps(new_test))

    # ─── Lọc features ──────────────────────────────────────────────────────
    used_items_sorted = sorted(used_items)
    feature_log: list[dict] = []
    for fname in args.features:
        fp = data_root / fname
        if not fp.exists():
            _log(f"Bỏ qua {fname} (không tìm thấy)")
            continue
        _log(f"Lọc feature {fname} theo old_ids ...")
        t0 = time.perf_counter()
        old_shape, new_shape = _filter_features(
            fp, used_items_sorted, fp, dry_run=False
        )
        feature_log.append(
            {"file": fname, "old_shape": list(old_shape), "new_shape": list(new_shape)}
        )
        _log(
            f"  {fname}: {old_shape} → {new_shape} "
            f"({time.perf_counter()-t0:.2f}s)"
        )

    # ─── Purge cache ───────────────────────────────────────────────────────
    purged: list[str] = []
    if args.purge_cache:
        for ext in ("*.pth", "*.npz"):
            for p in core_dir.glob(ext):
                # backup trước khi xóa (nhỏ — tiện rollback)
                shutil.copy2(p, backup_core / p.name)
                p.unlink()
                purged.append(p.name)
        # WEMA cache (nếu P2 đã tạo)
        for p in data_root.glob("wema_*.pth"):
            shutil.copy2(p, backup_dir / p.name)
            p.unlink()
            purged.append(p.name)
        if purged:
            _log(f"Đã xoá {len(purged)} cache file (kích thước cũ): {purged[:5]}...")

    # ─── Ghi metadata ──────────────────────────────────────────────────────
    metadata = {
        "timestamp": timestamp,
        "data_path": str(data_root),
        "core": args.core,
        "remap_users": args.remap_users,
        "n_users_old": n_users_old,
        "n_users_new": len(user_map) if user_map else n_users_old,
        "n_items_old": n_items_old,
        "n_items_new": len(item_map),
        "n_orphan_items_removed": n_orphan_items,
        "n_orphan_users_removed": n_orphan_users if args.remap_users else 0,
        "n_train_pairs_old": sum(len(v) for v in train.values()),
        "n_train_pairs_new": n_train_pairs,
        "n_val_pairs_old": sum(len(v) for v in val.values()),
        "n_val_pairs_new": n_val_pairs,
        "n_test_pairs_old": sum(len(v) for v in test.values()),
        "n_test_pairs_new": n_test_pairs,
        "feature_files": feature_log,
        "purged_cache_files": purged,
        "backup_dir": str(backup_dir),
        "item_map_sample_first10": dict(list(item_map.items())[:10]),
        "item_map_sample_last10": dict(list(item_map.items())[-10:]),
    }
    meta_path = data_root / "remap_metadata.json"
    # Lưu cả id maps đầy đủ riêng để có thể rollback chính xác
    maps_path = data_root / "remap_id_maps.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    maps_path.write_text(
        json.dumps(
            {
                "item_old_to_new": {str(k): v for k, v in item_map.items()},
                "user_old_to_new": (
                    {str(k): v for k, v in user_map.items()} if user_map else None
                ),
            }
        )
    )
    _log(f"Đã ghi metadata vào {meta_path.name} và {maps_path.name}")

    _log("─" * 64)
    _log("HOÀN TẤT.")
    _log("Bước tiếp theo:")
    _log(f"  1. python verify_remap.py --data-path {data_root}")
    _log("  2. Chạy huấn luyện 1 seed để so sánh NDCG@20 với baseline 0.0378")
    _log("─" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
