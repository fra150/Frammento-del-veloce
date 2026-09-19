"""Frammento del veloce - package src.

- frammento_2d : modello 2D toroidale (codice dr. Bulla Francesco)
- frammento_1d : modello 1D di riferimento (implementazione AI)
- demo_figure  : 7 figure dimostrative del preprint
"""

from .frammento_2d import Param, simula, riepilogo, invariante_nv, verifica_invarianza
from .frammento_1d import Params, FrammentoDelVeloce
from .frammento_gf import verifica_quiete, certifica_frammento, correggi_micro_errori, MemoriaGF, diagnostica_gf

__all__ = ["Param", "Params", "simula", "riepilogo", "invariante_nv", "verifica_invarianza", "FrammentoDelVeloce", "verifica_quiete", "certifica_frammento", "correggi_micro_errori", "MemoriaGF", "diagnostica_gf"]
__version__ = "1.0.0"
