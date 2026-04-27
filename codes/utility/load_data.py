"""
load_data.py — Data Loading & Graph Construction
==================================================

This module is responsible for:
  1. Reading train / val / test JSON split files.
  2. Building the sparse user-item interaction matrix  R  (n_users × n_items).
  3. Constructing the three graph structures required by MMHCL:
       • UI_mat   : user-item bipartite graph      (symmetrically normalised)
       • User_mat : user-user co-interaction graph  (row / random-walk normalised)
       • Item_mat : item-item multi-modal hypergraph (H · H^T, sym-normalised)
  4. Providing the BPR sampling function ``sample()`` used during training.

Data directory layout expected:
    data/<dataset>/
        5-core/
            train.json   — {uid: [iid, ...], ...}
            val.json
            test.json
        image_feat.npy   — (n_items, image_dim)  e.g. 4096-d from ResNet
        text_feat.npy    — (n_items, text_dim)   e.g. 768-d  from BERT
        audio_feat.npy   — (n_items, audio_dim)  [Tiktok only]

Graph construction pipeline for the Item-Item Hypergraph:
    For each modality m ∈ {image, text, [audio]}:
        1. Load feature matrix  F_m  ∈ R^{n_items × d_m}
        2. Compute cosine similarity:  S_m = norm(F_m) · norm(F_m)^T
        3. k-NN sparsification: keep only top-k values per row → A_m
    Concatenate:    H = [A_image | A_text | ...]      (the incidence matrix)
    Hypergraph:     Item_mat = H · H^T                (normalised by D^{-½})
"""

from __future__ import annotations

import argparse
import json
import random as rd
from time import time
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from utility.common import sparse_mx_to_torch_sparse_tensor as _sparse_mx_to_torch
from utility.norm import build_sim as _build_sim
from utility.parser import parse_args
from mmhcl_plus.topology.svd_augmentation import svd_filter_incidence

args: argparse.Namespace = parse_args()

import torch


def _torch_load(path: str) -> Any:
    """
    Compatibility wrapper for ``torch.load()``.

    PyTorch 2.6+ defaults to ``weights_only=True``, which breaks loading of
    sparse tensors and other pickled objects.  We explicitly set
    ``weights_only=False`` to maintain backward compatibility.
    """
    return torch.load(path, weights_only=False)


# ===========================================================================
#  Data — dataset loading, graph building, and BPR sampling
# ===========================================================================
class Data:
    """
    Loads a recommendation dataset and builds all required adjacency matrices.

    On first run, matrices are computed from raw features and cached as .pth
    files.  Subsequent runs load the cached versions for faster startup.

    Attributes:
        n_users     : dataset user count
        n_items     : dataset item count
        n_train     : number of training interactions
        n_test      : number of test interactions
        n_val       : number of validation interactions
        R           : scipy.sparse.dok_matrix — user-item interaction matrix (binary)
        train_items : training interactions per user
        test_set    : test interactions per user
        val_set     : validation interactions per user
        exist_users : user IDs that have at least one training interaction
    """

    def __init__(self, path: str, batch_size: int) -> None:
        """
        Args:
            path       : root path to the dataset (e.g. '../data/Clothing')
            batch_size : number of users to sample per training batch
        """
        self.path: str = f"{path}/{args.core}-core"
        self.batch_size: int = batch_size

        train_file: str = f"{path}/{args.core}-core/train.json"
        val_file: str = f"{path}/{args.core}-core/val.json"
        test_file: str = f"{path}/{args.core}-core/test.json"

        # --- Count users, items, and interactions from JSON files ---
        self.n_users: int = 0
        self.n_items: int = 0
        self.n_train: int = 0
        self.n_test: int = 0
        self.n_val: int = 0
        self.neg_pools: dict[int, list[int]] = {}

        self.exist_users: list[int] = []  # users with ≥ 1 training interaction

        train: dict[str, list[int]] = json.load(open(train_file))
        test: dict[str, list[int]] = json.load(open(test_file))
        val: dict[str, list[int]] = json.load(open(val_file))

        # Scan training data to determine n_users, n_items, and n_train
        for uid_str, items in train.items():
            if len(items) == 0:
                continue
            uid: int = int(uid_str)
            self.exist_users.append(uid)
            self.n_items = max(self.n_items, max(items))
            self.n_users = max(self.n_users, uid)
            self.n_train += len(items)

        # Scan test data
        for uid_str, items in test.items():
            uid = int(uid_str)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)
            except (TypeError, ValueError):
                continue

        # Scan validation data
        for uid_str, items in val.items():
            uid = int(uid_str)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_val += len(items)
            except (TypeError, ValueError):
                continue

        # +1 because IDs are 0-indexed
        self.n_items += 1
        self.n_users += 1

        # P1 (Acceleration Guide): refuse to load .pth graph caches whose tensor
        # shape no longer matches the current (n_users, n_items). Without this
        # guard, remap_clothing_orphans.py or any dataset re-split would
        # silently load the wrong-sized adjacency and crash deep inside
        # model.forward_plus with an opaque CUDA error. Best-effort: failures
        # never break dataloading.
        try:
            import glob as _p1_glob
            import os as _p1_os
            _p1_stale_dir = _p1_os.path.join(self.path, "_stale")
            for _p1_cache_path in _p1_glob.glob(_p1_os.path.join(self.path, "*.pth")):
                try:
                    _p1_obj = torch.load(
                        _p1_cache_path, weights_only=False, map_location="cpu"
                    )
                except Exception:
                    continue
                _p1_shape = getattr(_p1_obj, "shape", None)
                if _p1_shape is None:
                    continue
                _p1_base = _p1_os.path.basename(_p1_cache_path)
                _p1_expected = None
                if _p1_base.startswith("User_mat_"):
                    _p1_expected = (self.n_users, self.n_users)
                # NOTE: UI_mat / hypergraph caches have variable shapes
                # (bipartite, item×item, item×k*items) — we only assert the
                # tightly-typed user×user case. The rest get warned in logs.
                if _p1_expected and tuple(_p1_shape) != _p1_expected:
                    _p1_os.makedirs(_p1_stale_dir, exist_ok=True)
                    _p1_new_path = _p1_os.path.join(_p1_stale_dir, _p1_base)
                    _p1_os.replace(_p1_cache_path, _p1_new_path)
                    print(
                        f"[Data] P1 stale cache moved: {_p1_base}  "
                        f"shape={tuple(_p1_shape)} != expected={_p1_expected}"
                        "  →  _stale/"
                    )
        except Exception:
            pass

        self.print_statistics()

        # --- Build the binary user-item interaction matrix R ---
        # R[u, i] = 1 if user u interacted with item i in training
        self.R: sp.dok_matrix = sp.dok_matrix(
            (self.n_users, self.n_items), dtype=np.float32
        )
        self.R_Item_Interacts: sp.dok_matrix = sp.dok_matrix(
            (self.n_items, self.n_items), dtype=np.float32
        )

        self.train_items: dict[int, list[int]] = {}
        self.test_set: dict[int, list[int]] = {}
        self.val_set: dict[int, list[int]] = {}

        # Populate R and train_items from training data
        for uid_str, train_items_list in train.items():
            if len(train_items_list) == 0:
                continue
            uid = int(uid_str)
            for idx, i in enumerate(train_items_list):
                self.R[uid, i] = 1.0
            self.train_items[uid] = train_items_list

        # Populate test_set
        for uid_str, test_items_list in test.items():
            uid = int(uid_str)
            if len(test_items_list) == 0:
                continue
            try:
                self.test_set[uid] = test_items_list
            except (TypeError, ValueError, KeyError):
                continue

        # Populate val_set
        for uid_str, val_items_list in val.items():
            uid = int(uid_str)
            if len(val_items_list) == 0:
                continue
            try:
                self.val_set[uid] = val_items_list
            except (TypeError, ValueError, KeyError):
                continue

    # -----------------------------------------------------------------------
    #  Utility: scipy sparse → torch sparse
    # -----------------------------------------------------------------------
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx: sp.spmatrix) -> torch.Tensor:
        """Convert a scipy sparse matrix to a torch sparse COO tensor."""
        return _sparse_mx_to_torch(sparse_mx)

    def print_statistics(self) -> None:
        """Print basic dataset statistics to the console.

        Only emits output from the main process. When ``utility.batch_test`` is
        re-imported by a multiprocessing worker (e.g. ``Pool`` on Windows uses
        the ``spawn`` start method), the global ``data_generator = Data(...)``
        is rebuilt in every worker — without this guard each worker would
        race-print the same 3 stats lines to the parent's stdout pipe,
        producing the interleaved ``n_users=...n_train=...`` flood seen
        during evaluation epochs.
        """
        import multiprocessing as _mp
        if _mp.current_process().name != "MainProcess":
            return
        print(f"n_users={self.n_users}, n_items={self.n_items}")
        print(f"n_interactions={self.n_train + self.n_test}")
        sparsity = (self.n_train + self.n_test) / (self.n_users * self.n_items)
        print(f"n_train={self.n_train}, n_test={self.n_test}, sparsity={sparsity:.5f}")

    # ===================================================================
    #  BPR Sampling
    # ===================================================================
    def _ensure_sample_caches(self) -> None:
        """
        P5 (Acceleration Guide): pre-build per-user numpy arrays and Python sets.

        Built lazily on first ``sample()`` call so existing checkpoints / pickles
        stay backward-compatible. Memory: ~O(n_train) ints, negligible vs Item_mat.
        """
        if getattr(self, "_sample_caches_ready", False):
            return
        self._train_arr: dict[int, np.ndarray] = {
            u: np.asarray(items, dtype=np.int64)
            for u, items in self.train_items.items()
            if items
        }
        self._train_set: dict[int, set[int]] = {
            u: set(items) for u, items in self.train_items.items() if items
        }
        # Reusable RNG — np.random.default_rng is ~3x faster than np.random.randint
        self._sample_rng = np.random.default_rng()
        self._sample_caches_ready = True

    def sample(self) -> tuple[list[int], list[int], list[int]]:
        """
        Sample a batch of BPR (Bayesian Personalised Ranking) triplets.

        Returns:
            Tuple of (users, pos_items, neg_items), each a list of length batch_size.

        Vectorized version (P5):
          - Single call to rng.integers(0, n_items, size=(B, 8)) for negatives
            instead of B × while-True with per-call np.random.randint(size=1)
          - Per-user train_items pre-cached as numpy array + Python set
          - Empirical: 10.59 ms → 6.77 ms per batch on Amazon Clothing (B=2048)
        """
        self._ensure_sample_caches()

        # Sample batch_size users (without replacement if possible)
        users: list[int]
        if self.batch_size <= len(self.exist_users):
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        rng = self._sample_rng
        n_items_i = self.n_items
        # Oversample 8 negative candidates per user — 1-(1-sparsity)^8 ≈ 99.9999%
        # of users get at least one valid negative within the pre-drawn pool;
        # the rare miss falls through to the rejection-sample fallback below.
        cand = rng.integers(0, n_items_i, size=(self.batch_size, 8))

        pos_items: list[int] = [0] * self.batch_size
        neg_items: list[int] = [0] * self.batch_size
        for i, u in enumerate(users):
            arr = self._train_arr[u]
            train_set = self._train_set[u]
            # Single positive per user — uniform over user's interactions
            pos_items[i] = int(arr[rng.integers(0, len(arr))])
            # Reject from pre-drawn candidates first
            picked = False
            for c in cand[i]:
                ci = int(c)
                if ci not in train_set:
                    neg_items[i] = ci
                    picked = True
                    break
            if not picked:
                # Extremely rare fallback for very dense users
                while True:
                    ci = int(rng.integers(0, n_items_i))
                    if ci not in train_set:
                        neg_items[i] = ci
                        break
        return users, pos_items, neg_items

    # ===================================================================
    #  Adjacency Matrix Construction (legacy format from LightGCN/LATTICE)
    # ===================================================================
    def get_adj_mat(self) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        """
        Load or create the user-item adjacency matrix in three forms:
          - adj_mat      : raw bipartite adjacency (with self-loops)
          - norm_adj_mat : row-normalised (D^{-1} · (A + I))
          - mean_adj_mat : row-normalised (D^{-1} · A)

        These are cached as .npz files for fast reloading.
        Note: MMHCL uses ``get_UI_mat()`` instead; this method is kept for
        compatibility with the original LATTICE codebase.
        """
        try:
            t1: float = time()
            adj_mat: sp.spmatrix = sp.load_npz(self.path + "/s_adj_mat.npz")
            norm_adj_mat: sp.spmatrix = sp.load_npz(self.path + "/s_norm_adj_mat.npz")
            mean_adj_mat: sp.spmatrix = sp.load_npz(self.path + "/s_mean_adj_mat.npz")
            print("already load adj matrix", adj_mat.shape, time() - t1)
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + "/s_adj_mat.npz", adj_mat)
            sp.save_npz(self.path + "/s_norm_adj_mat.npz", norm_adj_mat)
            sp.save_npz(self.path + "/s_mean_adj_mat.npz", mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        """
        Build the (n_users + n_items) × (n_users + n_items) bipartite adjacency:
            A = [ 0   R  ]
                [ R^T 0  ]
        Then normalise it with D^{-1} (row normalisation).
        """
        t1: float = time()
        adj_mat: sp.spmatrix = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R: sp.lil_matrix = self.R.tolil()

        # Fill in the bipartite structure
        adj_mat[: self.n_users, self.n_users :] = R  # user → item
        adj_mat[self.n_users :, : self.n_users] = R.T  # item → user
        adj_mat = adj_mat.todok()
        print("already create adjacency matrix", adj_mat.shape, time() - t1)

        t2: float = time()

        def normalized_adj_single(adj: sp.spmatrix) -> sp.coo_matrix:
            """Row-normalise: D^{-1} · A"""
            rowsum: npt.NDArray[np.floating] = np.array(adj.sum(1))
            d_inv: npt.NDArray[np.floating] = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv: sp.dia_matrix = sp.diags(d_inv)
            norm_adj: sp.spmatrix = d_mat_inv.dot(adj)
            print("generate single-normalized adjacency matrix.")
            return norm_adj.tocoo()

        def get_D_inv(adj: sp.spmatrix) -> sp.dia_matrix:
            """Compute the inverse degree matrix D^{-1}."""
            rowsum: npt.NDArray[np.floating] = np.array(adj.sum(1))
            d_inv: npt.NDArray[np.floating] = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv: sp.dia_matrix = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj: sp.spmatrix) -> npt.NDArray[np.floating]:
            """Debug helper: verify normalisation by brute-force dense computation."""
            dense_A: npt.NDArray[np.floating] = np.array(adj.todense())
            degree: npt.NDArray[np.floating] = np.sum(dense_A, axis=1, keepdims=False)
            temp: npt.NDArray[np.floating] = np.dot(
                np.diag(np.power(degree, -1)), dense_A
            )
            print(
                "check normalized adjacency matrix whether equal to this laplacian matrix."
            )
            return temp

        # A + I  (self-loops) then normalise
        norm_adj_mat: sp.coo_matrix = normalized_adj_single(
            adj_mat + sp.eye(adj_mat.shape[0])
        )
        # A without self-loops, normalised
        mean_adj_mat: sp.coo_matrix = normalized_adj_single(adj_mat)

        print("already normalize adjacency matrix", time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    # ===================================================================
    #  Dense Normalisation Utility
    # ===================================================================
    def norm_dense(
        self, adj: torch.Tensor, normalization: str = "origin"
    ) -> torch.Tensor:
        """
        Normalise a dense adjacency matrix.

        Args:
            adj           : square adjacency matrix
            normalization :
                'sym'    : symmetric normalisation   D^{-½} A D^{-½}
                '2sym'   : asymmetric sym normalisation D_row^{-½} A D_col^{-½}
                'rw'     : random-walk normalisation D^{-1} A
                'origin' : no normalisation (return as-is)

        Returns:
            Normalised adjacency matrix (dense torch.Tensor).
        """
        L_norm: torch.Tensor

        if normalization == "sym":
            # Symmetric normalisation: D^{-1/2} · A · D^{-1/2}
            rowsum: torch.Tensor = torch.sum(adj, -1)
            d_inv_sqrt: torch.Tensor = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
            d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0.0
            d_mat_inv_sqrt: torch.Tensor = torch.diagflat(d_inv_sqrt)
            L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        elif normalization == "2sym":
            # Asymmetric symmetric normalisation (for non-square or directed):
            # D_row^{-1/2} · A · D_col^{-1/2}
            rowsum = torch.sum(adj, -1)
            d_row_inv_sqrt: torch.Tensor = torch.pow(rowsum, -0.5)
            d_row_inv_sqrt[torch.isinf(d_row_inv_sqrt)] = 0.0
            d_row_inv_sqrt[torch.isnan(d_row_inv_sqrt)] = 0.0
            d_row_mat_inv_sqrt: torch.Tensor = torch.diagflat(d_row_inv_sqrt)

            colsum: torch.Tensor = torch.sum(adj, -2)
            d_col_inv_sqrt: torch.Tensor = torch.pow(colsum, -0.5)
            d_col_inv_sqrt[torch.isinf(d_col_inv_sqrt)] = 0.0
            d_col_inv_sqrt[torch.isnan(d_col_inv_sqrt)] = 0.0
            d_col_mat_inv_sqrt: torch.Tensor = torch.diagflat(d_col_inv_sqrt)

            L_norm = torch.mm(torch.mm(d_row_mat_inv_sqrt, adj), d_col_mat_inv_sqrt)

        elif normalization == "rw":
            # Random-walk normalisation: D^{-1} · A
            rowsum = torch.sum(adj, -1)
            d_inv: torch.Tensor = torch.pow(rowsum, -1)
            d_inv[torch.isinf(d_inv)] = 0.0
            d_inv[torch.isnan(d_inv)] = 0.0
            d_mat_inv: torch.Tensor = torch.diagflat(d_inv)
            L_norm = torch.mm(d_mat_inv, adj)

        elif normalization == "origin":
            # No normalisation
            L_norm = adj

        return L_norm

    # ===================================================================
    #  User-Item Bipartite Graph
    # ===================================================================
    def get_UI_mat(self, norm_type: str = "sym") -> torch.Tensor:
        """
        Build or load the user-item bipartite adjacency matrix.

        Shape: (n_users + n_items) × (n_users + n_items), sparse.
        """
        print("Loading UI_mat:(" + norm_type + ")")
        t: float = time()
        UI_mat: torch.Tensor
        try:
            UI_mat = _torch_load(self.path + "/UI_mat_" + norm_type + ".pth")
        except Exception:
            # First run: build from scratch
            adj_mat_sp: sp.spmatrix = sp.dok_matrix(
                (self.n_users + self.n_items, self.n_users + self.n_items),
                dtype=np.float32,
            )
            adj_mat_lil: sp.lil_matrix = adj_mat_sp.tolil()
            R: sp.lil_matrix = self.R.tolil()
            adj_mat_lil[: self.n_users, self.n_users :] = R  # user → item
            adj_mat_lil[self.n_users :, : self.n_users] = R.T  # item → user
            adj_mat_dense: npt.NDArray[np.float32] = np.asarray(adj_mat_lil.todense())
            UI_mat = torch.from_numpy(adj_mat_dense).float()
            UI_mat = self.norm_dense(UI_mat, norm_type)
            UI_mat = UI_mat.to_sparse()
            torch.save(UI_mat, self.path + "/UI_mat_" + norm_type + ".pth")
        print("End Load UI_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return UI_mat

    def get_UI_single_mat(self, norm_type: str = "2sym") -> torch.Tensor:
        """
        Build or load the raw user-item interaction matrix R (NOT bipartite).

        Shape: (n_users × n_items), sparse.
        """
        print("Loading UI_single_mat:(" + norm_type + ")")
        t: float = time()
        UI_mat: torch.Tensor
        try:
            UI_mat = _torch_load(self.path + "/UI_single_mat_" + norm_type + ".pth")
        except Exception:
            adj_mat_dense: npt.NDArray[np.float32] = np.asarray(self.R.todense())
            UI_mat = torch.from_numpy(adj_mat_dense).float()
            UI_mat = self.norm_dense(UI_mat, norm_type)
            UI_mat = UI_mat.to_sparse()
            torch.save(UI_mat, self.path + "/UI_single_mat_" + norm_type + ".pth")
        print("End Load UI_single_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return UI_mat

    # ===================================================================
    #  User-User Co-Interaction Graph
    # ===================================================================
    def get_U2U_mat(self, norm_type: str = "rw") -> torch.Tensor:
        """
        Build or load the user-user co-interaction graph.

        Construction: User_mat = R · R^T,  diagonal zeroed out.
        Shape: (n_users × n_users), sparse.
        """
        print("Loading User_mat:(" + norm_type + ")")
        t: float = time()
        User_mat: torch.Tensor
        try:
            User_mat = _torch_load(self.path + "/User_mat_" + norm_type + ".pth")
        except Exception:
            R: torch.Tensor = torch.from_numpy(np.asarray(self.R.todense())).float()
            # Co-interaction: User_mat[u1, u2] = number of shared items
            User_mat = R @ R.T
            n_user: int = User_mat.size()[0]
            mask: torch.Tensor = torch.eye(n_user)
            User_mat[mask > 0] = 0  # Remove self-connections
            User_mat = self.norm_dense(User_mat, norm_type)
            User_mat = User_mat.to_sparse()
            torch.save(User_mat, self.path + "/User_mat_" + norm_type + ".pth")
        print("End Load User_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return User_mat

    # ===================================================================
    #  Item-Item Single-Modality Graphs (for ablation experiments)
    # ===================================================================
    def get_I2I_single_mat(
        self, norm_type: str = "sym"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | str]:
        """
        Build or load individual per-modality item-item similarity graphs.

        Returns:
            Tuple of (image_adj, text_adj, audio_adj_or_empty_string).
        """
        print("Loading I2I media-specific mat:(" + norm_type + ")")
        t: float = time()
        image_adj: torch.Tensor
        text_adj: torch.Tensor
        audio_adj: torch.Tensor
        try:
            image_adj = _torch_load(self.path + "/Image_mat_" + norm_type + ".pth")
            text_adj = _torch_load(self.path + "/Text_mat_" + norm_type + ".pth")
            if args.dataset == "tiktok":
                audio_adj = _torch_load(self.path + "/Audio_mat_" + norm_type + ".pth")
        except Exception:
            # Load raw multi-modal features
            image_feats: torch.Tensor = torch.tensor(
                np.load(f"../data/{args.dataset}/image_feat.npy")
            ).float()
            text_feats: torch.Tensor = torch.tensor(
                np.load(f"../data/{args.dataset}/text_feat.npy")
            ).float()
            if args.dataset == "tiktok":
                audio_feats: torch.Tensor = torch.tensor(
                    np.load(f"../data/{args.dataset}/audio_feat.npy")
                ).float()

            # Cosine similarity → k-NN → normalise for each modality
            image_adj = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            text_adj = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)
            if args.dataset == "tiktok":
                audio_adj = self.build_sim(audio_feats)
                audio_adj = self.build_knn_normalized_graph(audio_adj, topk=args.topk)

            image_adj = self.norm_dense(image_adj, norm_type)
            text_adj = self.norm_dense(text_adj, norm_type)
            if args.dataset == "tiktok":
                audio_adj = self.norm_dense(audio_adj, norm_type)

            # Convert to sparse and cache
            image_adj = image_adj.to_sparse()
            text_adj = text_adj.to_sparse()
            if args.dataset == "tiktok":
                audio_adj = audio_adj.to_sparse()

            torch.save(image_adj, self.path + "/Image_mat_" + norm_type + ".pth")
            torch.save(text_adj, self.path + "/Text_mat_" + norm_type + ".pth")
            if args.dataset == "tiktok":
                torch.save(audio_adj, self.path + "/Audio_mat_" + norm_type + ".pth")

        print(
            "End Load I2I media-specific mat:[%.1fs](" % (time() - t) + norm_type + ")"
        )
        if args.dataset == "tiktok":
            return image_adj, text_adj, audio_adj
        else:
            return image_adj, text_adj, ""

    # ===================================================================
    #  Item-Item Multi-Modal Hypergraph (2-modality: Clothing, Sports)
    # ===================================================================
    def get_I2I_Hypergrah_mat(self, norm_type: str = "origin") -> torch.Tensor:
        """
        Build the multi-modal hypergraph incidence matrix H (2 modalities).

        Returns:
            Sparse tensor of shape (n_items, 2 × n_items).
        """
        print(
            f"Loading I2I multi-media Hypergraph mat:({norm_type})_topk:{str(args.topk)}"
        )
        t: float = time()
        Hypergraph: torch.Tensor
        try:
            Hypergraph = _torch_load(
                f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth"
            )
        except Exception:
            # Load features for both modalities
            image_feats: torch.Tensor = torch.tensor(
                np.load(f"../data/{args.dataset}/image_feat.npy")
            ).float()
            text_feats: torch.Tensor = torch.tensor(
                np.load(f"../data/{args.dataset}/text_feat.npy")
            ).float()

            # Build per-modality k-NN similarity graphs
            image_adj: torch.Tensor = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            text_adj: torch.Tensor = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)

            # Concatenate along columns to form the hypergraph incidence matrix
            Hypergraph = torch.cat((image_adj, text_adj), dim=1)
            Hypergraph = self.norm_dense(Hypergraph, norm_type)
            Hypergraph = Hypergraph.to_sparse()
            torch.save(
                Hypergraph,
                f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth",
            )
        print(
            "End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t)
            + norm_type
            + ")"
        )
        return Hypergraph

    def get_I2I_Hypergraph_mul_mat(self, norm_type: str = "sym") -> torch.Tensor:
        """
        Build the final item-item multi-modal hypergraph:  H · H^T

        If args.use_svd_filtering is enabled, the incidence matrix H is
        SVD-filtered (top-K singular values zeroed) before multiplication,
        producing H̃ · H̃^T per Rev5.2 Section 2.2.

        Shape: (n_items × n_items), sparse.
        """
        svd_tag: str = (
            f"_svd{args.svd_top_k}" if getattr(args, "use_svd_filtering", 0) else ""
        )
        cache_name: str = (
            f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{args.topk}{svd_tag}.pth"
        )
        print(
            f"Loading I2I multi-media Hypergraph mul mat*mat.T:({norm_type})"
            f"_topk:{args.topk}{svd_tag}"
        )
        t: float = time()
        Hypergraph_mul: torch.Tensor
        try:
            Hypergraph_mul = _torch_load(cache_name)
        except Exception:
            Hypergraph: torch.Tensor = self.get_I2I_Hypergrah_mat("origin")

            # ── SVD Spectral Augmentation (Rev5.2) ──────────────────────
            if getattr(args, "use_svd_filtering", 0):
                print(
                    f"  Applying SVD filtering: zeroing top-{args.svd_top_k} "
                    "singular values of incidence matrix H"
                )
                H_filtered: torch.Tensor = svd_filter_incidence(
                    Hypergraph, top_k=args.svd_top_k
                )
                Hypergraph_mul = H_filtered @ H_filtered.T
                # Clamp to non-negative: SVD filtering introduces negative
                # entries which cause NaN in D^{-1/2} symmetric normalisation
                Hypergraph_mul = torch.clamp(Hypergraph_mul, min=0.0)
            else:
                Hypergraph_mul = torch.sparse.mm(
                    Hypergraph, Hypergraph.to_dense().T
                )
            # ────────────────────────────────────────────────────────────

            Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
            Hypergraph_mul = Hypergraph_mul.to_sparse()
            torch.save(Hypergraph_mul, cache_name)
        print(
            "End Load I2I multi-media Hypergraph mul mat*mat.T:[%.1fs](" % (time() - t)
            + norm_type
            + ")"
        )
        return Hypergraph_mul

    # ===================================================================
    #  PyTorch (.pt) feature variants (for datasets stored as .pt files)
    # ===================================================================
    def get_I2I_Hypergrah_mat_pt(self, norm_type: str = "origin") -> torch.Tensor:
        """Same as get_I2I_Hypergrah_mat() but loads features from .pt files."""
        print("Loading I2I multi-media Hypergraph mat:(" + norm_type + ")")
        t: float = time()
        Hypergraph: torch.Tensor
        try:
            Hypergraph = _torch_load(
                self.path + "/hypergraph_mat_" + norm_type + ".pth"
            )
        except Exception:
            image_feats: torch.Tensor = _torch_load(
                f"../data/{args.dataset}/img_feat.pt"
            )
            text_feats: torch.Tensor = _torch_load(
                f"../data/{args.dataset}/text_feat.pt"
            )

            image_adj: torch.Tensor = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            text_adj: torch.Tensor = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)

            Hypergraph = torch.cat((image_adj, text_adj), dim=1)
            Hypergraph = self.norm_dense(Hypergraph, norm_type)
            Hypergraph = Hypergraph.to_sparse()
            torch.save(Hypergraph, self.path + "/hypergraph_mat_" + norm_type + ".pth")
        print(
            "End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t)
            + norm_type
            + ")"
        )
        return Hypergraph

    def get_I2I_Hypergraph_mul_mat_pt(self, norm_type: str = "sym") -> torch.Tensor:
        """Same as get_I2I_Hypergraph_mul_mat() but uses .pt features."""
        svd_tag: str = (
            f"_svd{args.svd_top_k}" if getattr(args, "use_svd_filtering", 0) else ""
        )
        cache_name: str = (
            f"{self.path}/hypergraph_mat_mul{norm_type}{svd_tag}.pth"
        )
        print(
            f"Loading I2I multi-media Hypergraph mul mat*mat.T pytorch:"
            f"({norm_type}){svd_tag}"
        )
        t: float = time()
        Hypergraph_mul: torch.Tensor
        try:
            Hypergraph_mul = _torch_load(cache_name)
        except Exception:
            Hypergraph: torch.Tensor = self.get_I2I_Hypergrah_mat_pt("origin")

            if getattr(args, "use_svd_filtering", 0):
                print(
                    f"  Applying SVD filtering: zeroing top-{args.svd_top_k} "
                    "singular values of incidence matrix H"
                )
                H_filtered: torch.Tensor = svd_filter_incidence(
                    Hypergraph, top_k=args.svd_top_k
                )
                Hypergraph_mul = H_filtered @ H_filtered.T
                Hypergraph_mul = torch.clamp(Hypergraph_mul, min=0.0)
            else:
                Hypergraph_mul = torch.sparse.mm(
                    Hypergraph, Hypergraph.to_dense().T
                )

            Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
            Hypergraph_mul = Hypergraph_mul.to_sparse()
            torch.save(Hypergraph_mul, cache_name)
        print(
            "End Load I2I multi-media Hypergraph mul mat*mat.T pytorch:[%.1fs]("
            % (time() - t)
            + norm_type
            + ")"
        )
        return Hypergraph_mul

    # ===================================================================
    #  Item-Item Multi-Modal Hypergraph (3-modality: Tiktok)
    # ===================================================================
    def get_tiktok_I2I_Hypergrah_mat(self, norm_type: str = "origin") -> torch.Tensor:
        """
        Build the multi-modal hypergraph incidence matrix H for Tiktok
        (3 modalities: image, text, audio).

        Returns:
            Sparse tensor of shape (n_items, 3 × n_items).
        """
        print(
            f"Loading I2I multi-media Hypergraph mat:({norm_type})_topk:{str(args.topk)}"
        )
        t: float = time()
        Hypergraph: torch.Tensor
        try:
            Hypergraph = _torch_load(
                f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth"
            )
        except Exception:
            # Load all three modality features
            image_feats: torch.Tensor = torch.tensor(
                np.load(f"../data/{args.dataset}/image_feat.npy")
            ).float()
            text_feats: torch.Tensor = torch.tensor(
                np.load(f"../data/{args.dataset}/text_feat.npy")
            ).float()
            audio_feats: torch.Tensor = torch.tensor(
                np.load(f"../data/{args.dataset}/audio_feat.npy")
            ).float()

            # Build k-NN similarity graph for each modality
            image_adj: torch.Tensor = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            text_adj: torch.Tensor = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)

            audio_adj: torch.Tensor = self.build_sim(audio_feats)
            audio_adj = self.build_knn_normalized_graph(audio_adj, topk=args.topk)

            # Concatenate all three modalities into the incidence matrix
            Hypergraph = torch.cat(
                (torch.cat((image_adj, text_adj), dim=1), audio_adj), dim=1
            )
            Hypergraph = self.norm_dense(Hypergraph, norm_type)
            Hypergraph = Hypergraph.to_sparse()
            torch.save(
                Hypergraph,
                f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth",
            )
        print(
            "End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t)
            + norm_type
            + ")"
        )
        return Hypergraph

    def get_tiktok_I2I_Hypergraph_mul_mat(self, norm_type: str = "sym") -> torch.Tensor:
        """
        Build  H · H^T  for the 3-modality Tiktok hypergraph.

        Shape: (n_items × n_items), sparse.
        """
        svd_tag: str = (
            f"_svd{args.svd_top_k}" if getattr(args, "use_svd_filtering", 0) else ""
        )
        cache_name: str = (
            f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{args.topk}{svd_tag}.pth"
        )
        print(
            f"Loading I2I multi-media Hypergraph mul mat*mat.T:({norm_type})"
            f"_topk:{args.topk}{svd_tag}"
        )
        t: float = time()
        Hypergraph_mul: torch.Tensor
        try:
            Hypergraph_mul = _torch_load(cache_name)
        except Exception:
            Hypergraph: torch.Tensor = self.get_tiktok_I2I_Hypergrah_mat("origin")

            if getattr(args, "use_svd_filtering", 0):
                print(
                    f"  Applying SVD filtering: zeroing top-{args.svd_top_k} "
                    "singular values of incidence matrix H"
                )
                H_filtered: torch.Tensor = svd_filter_incidence(
                    Hypergraph, top_k=args.svd_top_k
                )
                Hypergraph_mul = H_filtered @ H_filtered.T
                Hypergraph_mul = torch.clamp(Hypergraph_mul, min=0.0)
            else:
                Hypergraph_mul = torch.sparse.mm(
                    Hypergraph, Hypergraph.to_dense().T
                )

            Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
            Hypergraph_mul = Hypergraph_mul.to_sparse()
            torch.save(Hypergraph_mul, cache_name)
        print(
            "End Load I2I multi-media Hypergraph mul mat*mat.T:[%.1fs](" % (time() - t)
            + norm_type
            + ")"
        )
        return Hypergraph_mul

    # ===================================================================
    #  Similarity & k-NN Graph Building Utilities
    # ===================================================================
    def build_sim(self, context: torch.Tensor) -> torch.Tensor:
        """Cosine similarity matrix (delegates to utility.norm.build_sim)."""
        return _build_sim(context)

    def build_knn_normalized_graph(self, adj: torch.Tensor, topk: int) -> torch.Tensor:
        """
        k-NN sparsification: keep only the top-k values per row.

        Args:
            adj  : (N, N) — dense similarity matrix
            topk : number of neighbours to keep

        Returns:
            (N, N) — sparse binary adjacency matrix with at most topk
                     non-zero entries per row.
        """
        knn_val: torch.Tensor
        knn_ind: torch.Tensor
        knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
        # Scatter top-k values into a zeroed matrix, then binarise
        adj = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        adj[adj > 0] = 1.0
        return adj


# ===========================================================================
#  P6 (Acceleration Guide): Async BPR prefetch — appended below class Data
# ===========================================================================


class AsyncBPRSampler:
    """
    P6 (Acceleration Guide): background-thread prefetcher for BPR triplets.

    Overlaps CPU sampling (~7 ms/batch on Clothing) with GPU forward+backward
    (~80-130 ms/batch on RTX 5090) using a producer/consumer queue.

    Empirically: 4-6% wall-clock saving per epoch with prefetch=2; higher on
    slower GPUs or smaller batch sizes.

    Usage
    -----
        sampler = AsyncBPRSampler(data_generator, prefetch=2)
        sampler.start()
        for _ in range(n_batch):
            users, pos, neg = sampler.sample()  # blocks if queue empty
            ...
        sampler.stop()  # signals worker, waits up to 2s

    Disable
    -------
        MMHCL_ASYNC_PREFETCH=0  → sample() falls through to sync data.sample()
        async_prefetch=False    → ditto via constructor

    Thread-safety
    -------------
    Single producer (worker) + single consumer (main) on queue.Queue is safe.
    The wrapped Data object's sample() is called only from the worker thread
    after start(), so there are no concurrent calls into Data.
    """

    def __init__(
        self,
        data_generator: "Data",
        prefetch: int = 2,
        async_prefetch: bool = True,
        logger=None,
    ) -> None:
        import os as _os
        import queue as _queue
        import threading as _threading

        self._dg = data_generator
        self._prefetch = max(1, int(prefetch))
        env_disabled = _os.environ.get("MMHCL_ASYNC_PREFETCH", "1") == "0"
        self._enabled = bool(async_prefetch) and not env_disabled
        self._logger = logger
        self._queue: _queue.Queue | None = None
        self._stop_evt: _threading.Event | None = None
        self._worker: _threading.Thread | None = None
        self._error: BaseException | None = None
        self._started = False

    # ------------------------------------------------------------------
    def start(self) -> "AsyncBPRSampler":
        """Spawn the worker thread. Idempotent."""
        if not self._enabled:
            self._log(
                "[AsyncBPRSampler] disabled (env or arg) — sample() will run sync"
            )
            return self
        if self._started:
            return self

        import queue as _queue
        import threading as _threading

        self._queue = _queue.Queue(maxsize=self._prefetch)
        self._stop_evt = _threading.Event()

        def _worker_loop() -> None:
            try:
                while not self._stop_evt.is_set():
                    batch = self._dg.sample()
                    # put with timeout so we re-check stop_evt periodically
                    while not self._stop_evt.is_set():
                        try:
                            self._queue.put(batch, timeout=0.5)
                            break
                        except Exception:
                            # Queue.Full → loop and recheck
                            continue
            except BaseException as exc:  # noqa: BLE001 — propagate cleanly
                self._error = exc
                # Sentinel so consumer wakes up
                try:
                    self._queue.put_nowait(None)
                except Exception:
                    pass

        self._worker = _threading.Thread(
            target=_worker_loop, name="bpr-prefetch", daemon=True
        )
        try:
            self._worker.start()
            self._started = True
            self._log(
                f"[AsyncBPRSampler] started: prefetch={self._prefetch} "
                "batches, daemon=True"
            )
        except Exception as exc:
            self._enabled = False
            self._log(
                f"[AsyncBPRSampler] start failed ({exc}); falling back to sync"
            )
        return self

    # ------------------------------------------------------------------
    def sample(self) -> tuple[list[int], list[int], list[int]]:
        """Get the next BPR batch. Blocks if queue is empty."""
        if not self._enabled or not self._started:
            return self._dg.sample()
        # Re-raise worker exception (if any) on the main thread
        if self._error is not None:
            err = self._error
            self._error = None
            raise err
        batch = self._queue.get()
        if batch is None:
            # Worker died — re-raise stored exception or fall back
            if self._error is not None:
                err = self._error
                self._error = None
                raise err
            return self._dg.sample()
        return batch

    # ------------------------------------------------------------------
    def stop(self, timeout: float = 2.0) -> None:
        """Signal the worker to stop and wait up to ``timeout`` seconds."""
        if not self._started or self._stop_evt is None:
            return
        self._stop_evt.set()
        # Drain queue so worker can exit its put loop
        try:
            while self._queue is not None and not self._queue.empty():
                self._queue.get_nowait()
        except Exception:
            pass
        if self._worker is not None:
            self._worker.join(timeout=timeout)
        self._started = False
        self._log("[AsyncBPRSampler] stopped")

    # ------------------------------------------------------------------
    def __enter__(self) -> "AsyncBPRSampler":
        return self.start()

    def __exit__(self, *exc_info) -> None:  # type: ignore[no-untyped-def]
        self.stop()

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self._logger is not None and hasattr(self._logger, "logging"):
            self._logger.logging(msg)
        else:
            print(msg, flush=True)



