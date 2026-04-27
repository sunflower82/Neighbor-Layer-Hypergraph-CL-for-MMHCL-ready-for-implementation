"""
rollback_remap.py — Hoàn tác P3 remap (lược orphan) cho Amazon Clothing
========================================================================

Script này phối hợp với 2 script trước đó:
  - remap_clothing_orphans.py — đã remap dataset (lược orphan items/users)
  - verify_remap.py — đã verify dataset sau remap

Khi nào cần rollback?
---------------------
  1. Sau khi chạy training thấy NDCG@20 lệch >0.5% so với baseline 0.0378
  2. Cần huấn luyện lại các phiên bản pre-remap để so sánh A/B
  3. Phát hiện bug trong dataset remap muốn quay về trạng thái gốc

Cơ chế khôi phục
----------------
remap_clothing_orphans.py luôn lưu backup vào:
    <data_path>/_orig_backup/<timestamp>/
        ├── 5-core/
        │   ├── train.json    (gốc)
        │   ├── val.json
        │   └── test.json
        │   └── *.pth          (cache đã xoá lúc remap)
        ├── image_feat.npy    (gốc)
        └── text_feat.npy     (gốc)

Thêm 2 file ở root data_path:
    remap_metadata.json    — tóm tắt
    remap_id_maps.json     — bidirectional ID maps (để rollback ngược)

Hai chế độ rollback
-------------------
  --mode backup   (mặc định, an toàn nhất)
      Khôi phục từ _orig_backup/<timestamp>/. Khớp y nguyên bản gốc.
      Yêu cầu: backup vẫn tồn tại (mặc định không bị xoá).

  --mode reverse-map
      Dùng remap_id_maps.json để map ngược new_id → old_id, ghi lại JSON
      và features. Hữu ích khi backup mất nhưng map vẫn còn.

Cách dùng
---------
    # Liệt kê các backup có sẵn
    python rollback_remap.py --data-path data/Clothing --list

    # Rollback từ backup mới nhất (mặc định)
    python rollback_remap.py --data-path data/Clothing

    # Rollback từ backup cụ thể
    python rollback_remap.py --data-path data/Clothing \\
        --backup-timestamp 20260427_091532

    # Rollback bằng reverse-map (nếu backup mất)
    python rollback_remap.py --data-path data/Clothing --mode reverse-map

    # Dry-run (chỉ in kế hoạch, không ghi)
    python rollback_remap.py --data-path data/Clothing --dry-run

Sau rollback
-------------
    python verify_remap.py --data-path data/Clothing
    (chỉ kỳ vọng pass nếu remap_metadata.json bị xóa cùng — verify sẽ thấy
     n_items lại là 23,033, không có warning continuity vì user gap đã quay lại)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# ===========================================================================
def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _list_backups(backup_root: Path) -> list[Path]:
    """Trả về danh sách thư mục backup, sorted theo thời gian giảm dần."""
    if not backup_root.exists():
        return []
    backups = [p for p in backup_root.iterdir() if p.is_dir()]
    backups.sort(reverse=True)
    return backups


# ===========================================================================
def cmd_list(data_root: Path) -> int:
    backup_root = data_root / "_orig_backup"
    backups = _list_backups(backup_root)
    if not backups:
        _log(f"Không có backup nào trong {backup_root}")
        return 1
    _log(f"Backup hiện có (mới nhất trước):")
    for i, b in enumerate(backups):
        size_mb = sum(f.stat().st_size for f in b.rglob("*") if f.is_file()) / 1e6
        files = list(b.rglob("*"))
        n_files = sum(1 for f in files if f.is_file())
        _log(
            f"  [{i}] {b.name}  ({size_mb:6.1f} MB, {n_files} files)"
        )
        # Show top-level contents
        for child in sorted(b.iterdir())[:5]:
            sub = "/" if child.is_dir() else ""
            _log(f"        - {child.name}{sub}")
    return 0


# ===========================================================================
def rollback_from_backup(
    data_root: Path,
    backup_dir: Path,
    core: int,
    dry_run: bool,
) -> int:
    """
    Khôi phục dataset từ thư mục backup. Trả về 0 nếu thành công.

    Trình tự:
      1. Verify backup có train.json, val.json, test.json
      2. Khôi phục splits → <core>-core/
      3. Khôi phục feature files (.npy) ở root
      4. Khôi phục cache .pth (nếu backup có) → <core>-core/
      5. Xoá remap_metadata.json + remap_id_maps.json
      6. Xoá WEMA cache cũ (nếu có) — sẽ build lại với shape gốc
    """
    core_dir = data_root / f"{core}-core"
    backup_core = backup_dir / f"{core}-core"

    # ─── Verify backup ────────────────────────────────────────────────────
    must_have = ["train.json", "val.json", "test.json"]
    missing = [f for f in must_have if not (backup_core / f).exists()]
    if missing:
        _log(f"Backup không hợp lệ — thiếu {missing} trong {backup_core}")
        return 1

    _log(f"Backup hợp lệ tại {backup_dir}")

    # ─── Plan list ────────────────────────────────────────────────────────
    actions: list[tuple[str, Path, Path]] = []  # (verb, src, dst)

    # JSON splits
    for fname in must_have:
        actions.append(("restore-json", backup_core / fname, core_dir / fname))

    # Cache .pth trong backup_core (lúc remap đã backup trước khi xoá)
    for src in backup_core.glob("*.pth"):
        actions.append(("restore-pth", src, core_dir / src.name))

    # Feature files ở root
    for npy_path in backup_dir.glob("*.npy"):
        actions.append(("restore-npy", npy_path, data_root / npy_path.name))

    # Files cần xoá (post-remap artifacts)
    deletions: list[Path] = []
    for f in ("remap_metadata.json", "remap_id_maps.json"):
        p = data_root / f
        if p.exists():
            deletions.append(p)
    # WEMA cache có thể đang dùng shape sai
    deletions.extend(data_root.glob("wema_*.pth"))
    # _stale dir (P1 đã move) — clear vì giờ shape đã khớp lại
    stale_dir = core_dir / "_stale"
    if stale_dir.exists():
        deletions.append(stale_dir)

    # ─── Print plan ───────────────────────────────────────────────────────
    _log("─" * 64)
    _log(f"Kế hoạch rollback ({len(actions)} restore + {len(deletions)} xoá):")
    for verb, src, dst in actions:
        _log(f"  [{verb}] {src.name} → {dst.relative_to(data_root)}")
    for p in deletions:
        _log(f"  [delete] {p.relative_to(data_root)}")

    if dry_run:
        _log("DRY-RUN: không ghi gì.")
        return 0

    # ─── Save current (post-remap) state into a quarantine dir ────────────
    quarantine = data_root / "_pre_rollback" / datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine.mkdir(parents=True, exist_ok=True)
    quarantine_core = quarantine / f"{core}-core"
    quarantine_core.mkdir()
    _log(f"Backup trạng thái hiện tại (post-remap) vào {quarantine}")
    for fname in must_have:
        cur = core_dir / fname
        if cur.exists():
            shutil.copy2(cur, quarantine_core / fname)
    for npy in data_root.glob("*.npy"):
        shutil.copy2(npy, quarantine / npy.name)
    for f in ("remap_metadata.json", "remap_id_maps.json"):
        cur = data_root / f
        if cur.exists():
            shutil.copy2(cur, quarantine / f)

    # ─── Execute restore ──────────────────────────────────────────────────
    for verb, src, dst in actions:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        _log(f"  ✓ {verb}: {src.name}")

    # ─── Execute delete ───────────────────────────────────────────────────
    for p in deletions:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        _log(f"  ✓ delete: {p.name}")

    _log("─" * 64)
    _log("ROLLBACK HOÀN TẤT.")
    _log(f"Trạng thái post-remap được lưu tại {quarantine}")
    _log("Bước tiếp theo:")
    _log(f"  1. python verify_remap.py --data-path {data_root}")
    _log("     (sẽ in cảnh báo về user gap — bình thường, đó là dữ liệu gốc)")
    _log("  2. Nếu OK, có thể xoá thủ công _orig_backup/ và _pre_rollback/")
    return 0


# ===========================================================================
def rollback_via_reverse_map(
    data_root: Path,
    core: int,
    dry_run: bool,
) -> int:
    """
    Khôi phục bằng cách áp ngược remap_id_maps.json. Dùng khi backup mất.

    Lưu ý: chỉ khôi phục được những item/user còn được xuất hiện trong split
    sau remap. Item orphan đã loại trước remap KHÔNG quay lại — chúng vẫn
    không có interaction nào nên không cần.
    """
    maps_p = data_root / "remap_id_maps.json"
    if not maps_p.exists():
        _log(f"Không tìm thấy {maps_p} — không thể reverse-map")
        return 1

    maps = json.loads(maps_p.read_text())
    item_old_to_new: dict[int, int] = {
        int(k): v for k, v in maps["item_old_to_new"].items()
    }
    item_new_to_old: dict[int, int] = {v: k for k, v in item_old_to_new.items()}

    user_old_to_new = maps.get("user_old_to_new")
    user_new_to_old: dict[int, int] | None = None
    if user_old_to_new:
        u_o2n: dict[int, int] = {int(k): v for k, v in user_old_to_new.items()}
        user_new_to_old = {v: k for k, v in u_o2n.items()}

    _log(
        f"Reverse map: {len(item_new_to_old):,} items, "
        f"{len(user_new_to_old) if user_new_to_old else 0:,} users"
    )

    core_dir = data_root / f"{core}-core"
    splits = ["train.json", "val.json", "test.json"]

    # Build new split dicts (with old IDs)
    plan: dict[str, dict[str, list[int]]] = {}
    for fname in splits:
        cur = core_dir / fname
        if not cur.exists():
            _log(f"Thiếu {cur}")
            return 1
        data = json.loads(cur.read_text())
        out: dict[str, list[int]] = {}
        for k_str, items in data.items():
            new_uid = int(k_str)
            old_uid = (
                user_new_to_old[new_uid]
                if user_new_to_old and new_uid in user_new_to_old
                else new_uid
            )
            old_items = [item_new_to_old[int(i)] for i in items]
            out[str(old_uid)] = old_items
        plan[fname] = out
        _log(
            f"  {fname}: {len(out):,} users, "
            f"{sum(len(v) for v in out.values()):,} pairs (đã reverse-map)"
        )

    if dry_run:
        _log("DRY-RUN: không ghi gì.")
        return 0

    # Quarantine current state
    quarantine = data_root / "_pre_rollback" / datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine.mkdir(parents=True, exist_ok=True)
    (quarantine / f"{core}-core").mkdir()
    for fname in splits:
        shutil.copy2(core_dir / fname, quarantine / f"{core}-core" / fname)
    _log(f"Quarantine trạng thái hiện tại tại {quarantine}")

    # Write reversed splits
    for fname, data in plan.items():
        (core_dir / fname).write_text(json.dumps(data))
        _log(f"  ✓ wrote {fname}")

    # Reverse feature files
    # Item map: new_id i corresponds to old_id item_new_to_old[i]
    # To restore: insert each row into old_max + 1 sized array
    n_items_old = max(item_new_to_old.values()) + 1
    for npy_path in data_root.glob("*.npy"):
        if not npy_path.name.endswith(("image_feat.npy", "text_feat.npy")):
            continue
        arr = np.load(npy_path)
        if arr.shape[0] != len(item_new_to_old):
            _log(
                f"  Bỏ qua {npy_path.name}: shape[0]={arr.shape[0]} "
                f"!= n_items_new={len(item_new_to_old)} "
                "(có thể chưa filter sau remap?)"
            )
            continue
        new_arr = np.zeros((n_items_old,) + arr.shape[1:], dtype=arr.dtype)
        for new_id, old_id in item_new_to_old.items():
            new_arr[old_id] = arr[new_id]
        # Quarantine current
        shutil.copy2(npy_path, quarantine / npy_path.name)
        np.save(npy_path, new_arr)
        _log(
            f"  ✓ {npy_path.name}: {arr.shape} → {new_arr.shape}  "
            "(orphan rows = zeros)"
        )
        _log(
            "    Lưu ý: orphan items được fill bằng 0 — không khớp byte với "
            "feature gốc. Nếu cần đúng từng byte, dùng --mode backup."
        )

    # Cleanup post-remap artifacts
    for f in ("remap_metadata.json", "remap_id_maps.json"):
        p = data_root / f
        if p.exists():
            shutil.copy2(p, quarantine / f)
            p.unlink()
            _log(f"  ✓ delete: {f}")

    for cache in data_root.glob("wema_*.pth"):
        cache.unlink()
        _log(f"  ✓ delete: {cache.name}")

    stale_dir = core_dir / "_stale"
    if stale_dir.exists():
        shutil.rmtree(stale_dir)
        _log("  ✓ delete: 5-core/_stale/")

    _log("─" * 64)
    _log("REVERSE-MAP ROLLBACK HOÀN TẤT.")
    _log(f"Trạng thái post-remap quarantine: {quarantine}")
    return 0


# ===========================================================================
def main() -> int:
    p = argparse.ArgumentParser(
        description="Hoàn tác P3 remap orphan cho Amazon Clothing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path", type=str, default="data/Clothing")
    p.add_argument("--core", type=int, default=5)
    p.add_argument(
        "--mode",
        choices=["backup", "reverse-map"],
        default="backup",
        help="backup = khôi phục từ _orig_backup (chính xác từng byte). "
        "reverse-map = dùng remap_id_maps.json (orphan fill zeros).",
    )
    p.add_argument(
        "--backup-timestamp",
        type=str,
        default=None,
        help="Backup cụ thể (YYYYMMDD_HHMMSS). Mặc định: mới nhất.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Chỉ liệt kê backup hiện có, không rollback",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_path).resolve()
    if not data_root.exists():
        _log(f"Data path không tồn tại: {data_root}")
        return 1

    if args.list:
        return cmd_list(data_root)

    if args.mode == "reverse-map":
        return rollback_via_reverse_map(data_root, args.core, args.dry_run)

    # mode = backup
    backup_root = data_root / "_orig_backup"
    backups = _list_backups(backup_root)
    if not backups:
        _log(
            f"Không có backup trong {backup_root}. Thử --mode reverse-map "
            "nếu remap_id_maps.json còn."
        )
        return 1

    if args.backup_timestamp:
        chosen = backup_root / args.backup_timestamp
        if not chosen.exists():
            _log(f"Backup không tồn tại: {chosen}")
            _log("Liệt kê có sẵn: python rollback_remap.py --list ...")
            return 1
    else:
        chosen = backups[0]
        _log(f"Sử dụng backup mới nhất: {chosen.name}")

    return rollback_from_backup(data_root, chosen, args.core, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
