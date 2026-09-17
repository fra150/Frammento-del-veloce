"""Frammento del veloce - package src.

- frammento_2d : modello 2D toroidale (codice dr. Bulla Francesco)
- frammento_1d : modello 1D di riferimento (implementazione AI)
- demo_figure  : 7 figure dimostrative del preprint
"""

from .frammento_2d import Param, simula, riepilogo, invariante_nv, verifica_invarianza
from .frammento_1d import Params, FrammentoDelVeloce

__all__ = ["Param", "Params", "simula", "riepilogo", "invariante_nv", "verifica_invarianza", "FrammentoDelVeloce"]
__version__ = "1.0.0"
