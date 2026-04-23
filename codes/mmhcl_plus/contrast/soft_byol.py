from __future__ import annotations

import logging
import os
import sys
import warnings

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def _maybe_torch_compile(fn):
    """Try to wrap `fn` with `torch.compile`, falling back gracefully.

    The fallback path is now observable: we emit a one-shot warning (both via
    `warnings.warn` and the module logger) explaining *why* compilation was
    skipped, so "silent" no-compile environments are no longer silent during
    debugging. The boolean environment variable `MMHCL_DISABLE_TORCH_COMPILE`
    still provides an explicit opt-out, and sets a different (quiet) log level.
    """
    if os.environ.get("MMHCL_DISABLE_TORCH_COMPILE", "").strip():
        logger.info(
            "[soft_byol] torch.compile disabled via MMHCL_DISABLE_TORCH_COMPILE; "
            "using eager %s.", fn.__name__,
        )
        return fn
    if sys.platform == "win32":
        msg = (
            f"[soft_byol] torch.compile skipped on Windows "
            f"(Triton inductor backend unavailable); using eager {fn.__name__}."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)
        return fn
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        msg = (
            f"[soft_byol] torch.compile missing (torch={torch.__version__}); "
            f"using eager {fn.__name__}."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)
        return fn
    try:
        return compile_fn(fn, dynamic=True, fullgraph=False)
    except Exception as exc:  # noqa: BLE001 - report any compile failure
        msg = (
            f"[soft_byol] torch.compile failed ({type(exc).__name__}: {exc}); "
            f"falling back to eager {fn.__name__}."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)
        return fn


def _soft_byol_alignment_impl(
    online: torch.Tensor,
    target: torch.Tensor,
    soft_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    online = F.normalize(online, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    sim = (online * target).sum(dim=-1)
    loss = 2.0 - 2.0 * sim
    if soft_weights is not None:
        loss = loss * soft_weights
    return loss.mean()


soft_byol_alignment = _maybe_torch_compile(_soft_byol_alignment_impl)
