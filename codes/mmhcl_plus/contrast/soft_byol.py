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

    Windows handling
    ----------------
    Earlier revisions hard-blocked Windows because the Triton Inductor backend
    was upstream-Linux-only.  As of Triton ≥ 3.3 (PyTorch ≥ 2.7, CUDA ≥ 12.8),
    the official ``triton-windows`` package ships SM120 Blackwell support, so
    we now probe ``import triton`` instead of the bare platform string:

        - ``triton-windows`` installed  →  allow ``torch.compile``
        - Windows + no triton           →  fall back to eager (with warning)
        - Non-Windows                   →  unchanged behaviour

    This mirrors the recipe in
    ``mmhcl_rev52_speedup_analysis_EN_implementation.tex`` §Step 3 and unlocks
    Inductor kernel fusion for ``vicreg_loss`` and ``soft_byol_alignment`` on
    the RTX 5090 / Windows 11 LTSC stack.
    """
    if os.environ.get("MMHCL_DISABLE_TORCH_COMPILE", "").strip():
        logger.info(
            "[soft_byol] torch.compile disabled via MMHCL_DISABLE_TORCH_COMPILE; "
            "using eager %s.", fn.__name__,
        )
        return fn
    if sys.platform == "win32":
        try:
            import triton  # noqa: F401 - probe for triton-windows availability
        except ImportError:
            msg = (
                f"[soft_byol] torch.compile skipped on Windows "
                f"(triton-windows not installed; `pip install triton-windows` "
                f"to unlock Inductor kernel fusion); using eager {fn.__name__}."
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
        compiled = compile_fn(fn, dynamic=True, fullgraph=False)
    except Exception as exc:  # noqa: BLE001 - report any compile failure
        msg = (
            f"[soft_byol] torch.compile failed ({type(exc).__name__}: {exc}); "
            f"falling back to eager {fn.__name__}."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)
        return fn

    # ── Belt-and-suspenders: catch FIRST-CALL Inductor failures ──────────────
    # ``torch.compile`` defers actual codegen + C++ build until the first
    # invocation, so decoration-time try/except above does NOT catch the
    # ``CppCompileError`` raised when MSVC ``cl.exe`` chokes on a temp path
    # containing a space (e.g. ``torchinductor_Anh Khoi``).  Wrap the compiled
    # callable so that the first runtime failure permanently degrades to eager
    # for the remainder of the run; subsequent calls bypass the failed compile.
    _state: dict[str, bool] = {"degraded": False}

    def _resilient(*args, **kwargs):
        if _state["degraded"]:
            return fn(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - record any compile/runtime failure
            _state["degraded"] = True
            short = str(exc)
            if len(short) > 240:
                short = short[:240] + "..."
            msg = (
                f"[soft_byol] torch.compile runtime failure for {fn.__name__} "
                f"({type(exc).__name__}: {short}); falling back to eager for "
                f"the remainder of the run."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            logger.warning(msg)
            return fn(*args, **kwargs)

    _resilient.__name__ = fn.__name__
    _resilient.__qualname__ = getattr(fn, "__qualname__", fn.__name__)
    _resilient.__doc__ = fn.__doc__
    _resilient.__wrapped__ = fn  # type: ignore[attr-defined]
    return _resilient


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
