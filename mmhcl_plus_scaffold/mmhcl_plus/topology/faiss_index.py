import torch

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

class ANNIndex:
    def __init__(self, dim: int, metric: str = 'ip', backend: str = 'torch'):
        self.dim = dim
        self.metric = metric
        self.backend = backend if backend != 'faiss' or HAS_FAISS else 'torch'
        self._xb = None
        self._faiss = None

    def build(self, xb: torch.Tensor):
        xb = xb.detach().cpu().float().contiguous()
        self._xb = xb
        if self.backend == 'faiss':
            if self.metric == 'ip':
                self._faiss = faiss.IndexFlatIP(self.dim)
            else:
                self._faiss = faiss.IndexFlatL2(self.dim)
            self._faiss.add(xb.numpy())
        return self

    def search(self, xq: torch.Tensor, k: int = 32):
        xq_cpu = xq.detach().cpu().float().contiguous()
        if self.backend == 'faiss' and self._faiss is not None:
            sim, idx = self._faiss.search(xq_cpu.numpy(), k)
            return torch.from_numpy(sim), torch.from_numpy(idx)
        sim = xq_cpu @ self._xb.T
        topv, topi = torch.topk(sim, k=min(k, self._xb.size(0)), dim=-1)
        return topv, topi
