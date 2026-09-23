"""Frammento del veloce - package src.

- frammento_2d : modello 2D toroidale (codice dr. Bulla Francesco)
- frammento_1d : modello 1D di riferimento (implementazione AI)
- frammento_gf : quiete + certificazione + memoria
- confronto_bio: coerenza interna LFP/theta-gamma + confronto spettrale onesto
- demo_figure  : 7 figure dimostrative del preprint
"""

from .frammento_2d import Param, simula, riepilogo, invariante_nv, verifica_invarianza
from .frammento_1d import Params, FrammentoDelVeloce
from .frammento_gf import verifica_quiete, certifica_frammento, correggi_micro_errori, MemoriaGF, diagnostica_gf
from .confronto_bio import valida_sistema_sintetico, confronta_sintetico_vs_reale, carica_eeg_csv
from .rete_frammento import (
    ReteFrammento, ReteIngenuaCondivisa, esegui_test_1000, genera_cue,
    cue_parziale, ricostruisci_associativo, esegui_test_associativo,
    salva_report_assoc)
from .fase15 import (
    ReteReplay, ReteEWC, esegui_confronto_cl, sweep_pareto_cl,
    misura_plasticita, seleziona_prior, test_ood_mix_retrieval,
    test_ood_rumore_retrieval)
from .fase16 import (
    sonno, distilla_gist, valuta_sonno_mix, valuta_sonno_rumore,
    evici_slot, esegui_eviction_study, misura_transfer,
    seleziona_prior_validato, seleziona_prior_coarse_to_fine,
    valuta_retrieval_repair)

__all__ = ["Param", "Params", "simula", "riepilogo", "invariante_nv", "verifica_invarianza", "FrammentoDelVeloce", "verifica_quiete", "certifica_frammento", "correggi_micro_errori", "MemoriaGF", "diagnostica_gf", "valida_sistema_sintetico", "confronta_sintetico_vs_reale", "carica_eeg_csv", "ReteFrammento", "ReteIngenuaCondivisa", "esegui_test_1000", "genera_cue", "cue_parziale", "ricostruisci_associativo", "esegui_test_associativo", "salva_report_assoc", "ReteReplay", "ReteEWC", "esegui_confronto_cl", "sweep_pareto_cl", "misura_plasticita", "seleziona_prior", "test_ood_mix_retrieval", "test_ood_rumore_retrieval", "sonno", "distilla_gist", "valuta_sonno_mix", "valuta_sonno_rumore", "evici_slot", "esegui_eviction_study", "misura_transfer", "seleziona_prior_validato", "seleziona_prior_coarse_to_fine", "valuta_retrieval_repair"]
__version__ = "1.0.0"
