"""
ANN index for soft-topology / Jaccard neighbour sets (MMHCL+ Optimization Report — Opt. 3).

Uses FAISS GPU IndexFlatIP when available for faster batched search vs brute-force torch.topk.
"""

from __future__ import annotations

import torch

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False


def _faiss_num_gpus() -> int:
    if not HAS_FAISS:
        return 0
    try:
        return int(faiss.get_num_gpus())
    except Exception:
        return 0


class ANNIndex:
    def __init__(
        self,
        dim: int,
        metric: str = "ip",
        backend: str = "torch",
        *,
        use_gpu: bool = True,
        fp16_store: bool = True,
    ):
        self.dim = dim
        self.metric = metric
        self.backend = backend if backend != "faiss" or HAS_FAISS else "torch"
        self.use_gpu = (
            bool(use_gpu) and self.backend == "faiss" and _faiss_num_gpus() > 0
        )
        self.fp16_store = bool(fp16_store)
        self._xb: torch.Tensor | None = None
        self._faiss = None

    def build(self, xb: torch.Tensor) -> ANNIndex:
        xb = xb.detach().float().contiguous()
        self._xb = xb
        if self.backend != "faiss" or not HAS_FAISS:
            return self

        arr = xb.numpy()
        if self.metric != "ip":
            self._faiss = faiss.IndexFlatL2(self.dim)
            self._faiss.add(arr)
            return self

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            try:
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = self.fp16_store
                self._faiss = faiss.GpuIndexFlatIP(res, self.dim, cfg)
            except Exception:
                cpu_ip = faiss.IndexFlatIP(self.dim)
                try:
                    self._faiss = faiss.index_cpu_to_gpu(res, 0, cpu_ip)
                except Exception:
                    self._faiss = cpu_ip
            self._faiss.add(arr)
        else:
            self._faiss = faiss.IndexFlatIP(self.dim)
            self._faiss.add(arr)
        return self

    def search(
        self, xq: torch.Tensor, k: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev = xq.device
        xq_cpu = xq.detach().float().contiguous()
        k_eff = min(k, self._xb.size(0)) if self._xb is not None else k

        if self.backend == "faiss" and self._faiss is not None:
            sim, idx = self._faiss.search(xq_cpu.numpy(), k_eff)
            return (
                torch.from_numpy(sim).to(dev),
                torch.from_numpy(idx).long().to(dev),
            )

        assert self._xb is not None
        sim = xq_cpu @ self._xb.T
        topv, topi = torch.topk(sim, k=k_eff, dim=-1)
        return topv.to(dev), topi.to(dev)
