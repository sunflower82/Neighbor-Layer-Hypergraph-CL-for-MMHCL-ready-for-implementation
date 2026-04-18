"""
MMHCL+ diagnostic utilities --- Revision 5.2.

These modules are *optional* instrumentation used for sanity checks
described in ``mmhcl_plus_ablation_guide_full_translation.tex``:

  * :mod:`mmhcl_plus.diagnostics.spectral_radius`
      Tracks the spectral radius of the hypergraph diffusion operator
      ``Theta = I - L`` across training epochs.  The design note in
      Rev5.2 expects ``lambda_max(Theta)`` to stay below 1 and to
      decrease when Dirichlet regularisation is enabled.

A Mutual-Information estimator is left as a future extension (see
the guide for MINE / KSG suggestions).  The spectral-radius tracker
is self-contained and has no GPU dependencies beyond NumPy / SciPy.
"""
from __future__ import annotations

from .spectral_radius import (
    SpectralRadiusTracker,
    compute_spectral_gap,
    compute_spectral_radius,
    track_spectral_radius_per_epoch,
)

__all__ = [
    "SpectralRadiusTracker",
    "compute_spectral_gap",
    "compute_spectral_radius",
    "track_spectral_radius_per_epoch",
]
