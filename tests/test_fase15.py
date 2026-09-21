"""Fase 15: baseline CL forti + plasticita' + retrieval OOD (fast, N=16).

Tutti deterministici (stocastico=False). Nessun file pesante, nessun download.
"""

import numpy as np

from src.fase15 import (
    ReteEWC,
    ReteReplay,
    esegui_confronto_cl,
    misura_plasticita,
    salva_report_cl,
    salva_report_ood,
    salva_report_plasticita,
    seleziona_prior,
    sweep_pareto_cl,
    test_ood_mix_retrieval as ood_mix_retrieval,
    test_ood_rumore_retrieval as ood_rumore_retrieval,
)
from src.rete_frammento import ReteFrammento, cue_parziale, genera_cue


def test_cl_shift_separa_metodi():
    # con lo shift la condivisa DEVE degradare, la protetta no (anti-vacuita')
    r = esegui_confronto_cl(n_cert=10, n_nuove=20, N=16, T=0.05, seed=7,
                            shift=True)
    assert r["n_certificati"] > 0
    assert r["protetta"]["distrutti"] == 0
    assert r["protetta"]["max_degrado"] <= 1e-9
    assert r["ingenua"]["max_degrado"] > 1e-6
    assert r["ingenua"]["distrutti"] > 0


def test_ewc_lam0_coincide_ingenua():
    # sanita': lam=0 ed eta=0.1 -> stessa dinamica dell'ingenua
    r = esegui_confronto_cl(n_cert=10, n_nuove=20, N=16, T=0.05, seed=7,
                            shift=True, ewc_lam=0.0, ewc_eta=0.1)
    assert abs(r["ewc"]["max_degrado"] - r["ingenua"]["max_degrado"]) < 1e-12
    assert r["ewc"]["distrutti"] == r["ingenua"]["distrutti"]


def test_ewc_riduce_forgetting():
    # EWC-lite ben tarato riduce il forgetting senza bloccare B
    r = esegui_confronto_cl(n_cert=10, n_nuove=20, N=16, T=0.05, seed=7,
                            shift=True, ewc_lam=0.5, ewc_eta=0.1)
    assert r["ewc"]["max_degrado"] < r["ingenua"]["max_degrado"]
    assert r["ewc"]["distrutti"] <= r["ingenua"]["distrutti"]
    # plasticita' su B non crollata (impara ancora il nuovo set)
    assert r["ewc"]["q_nuove"] >= r["ingenua"]["q_nuove"] - 0.05


def test_replay_buffer_grande_aiuta():
    # buffer che trattiene tutto dimentica meno del FIFO corto
    r_pic = esegui_confronto_cl(n_cert=10, n_nuove=20, N=16, T=0.05, seed=7,
                                shift=True, replay_k=10, buffer_max=5)
    r_big = esegui_confronto_cl(n_cert=10, n_nuove=20, N=16, T=0.05, seed=7,
                                shift=True, replay_k=10, buffer_max=200)
    assert (r_big["replay"]["degrado_medio"]
            <= r_pic["replay"]["degrado_medio"] + 1e-12)


def test_plasticita_lineare_e_piatta():
    righe = misura_plasticita(n_list=[10, 20], N=16, T=0.05, seed=7)
    assert righe[0]["tasso_cert"] == 1.0
    assert righe[1]["tasso_cert"] == 1.0
    # memoria lineare: raddoppia con n
    assert righe[1]["memoria_bytes"] == 2 * righe[0]["memoria_bytes"]
    # Q piatta, costo per ricordo contenuto
    assert abs(righe[1]["q_media"] - righe[0]["q_media"]) < 0.02
    assert righe[1]["tempo_per_ricordo_ms"] < 50.0


def test_rifiuta_copertura_e_qualita():
    # memoria limitata: copertura bassa ma Q dei memorizzati intatta
    cue = genera_cue(8, seed=7)
    r = ReteFrammento(N=16, T=0.05, capacita_max=3, politica="rifiuta")
    for c in cue:
        r.impara(c)
    assert len(r.slot) <= 3
    qs = [v["q_cert"] for v in r.slot.values()]
    assert float(np.mean(qs)) > 0.40  # sopra soglia gf


def test_retrieval_esatto_e_fragile():
    # in-distribution: retrieval perfetto; con rumore forte crolla (onesto)
    righe = ood_rumore_retrieval(N=16, T=0.05, seed=7,
                                 rumori=(0.0, 2.0))
    assert righe[0]["accuratezza"] == 1.0
    assert righe[1]["accuratezza"] < 1.0


def test_mix_estremi_recuperano_lato_giusto():
    righe = ood_mix_retrieval(N=16, T=0.05, seed=7,
                              alpha_list=(0.0, 1.0))
    assert righe[0]["vincitore"] == "B"  # alpha=0 -> puro B
    assert righe[1]["vincitore"] == "A"  # alpha=1 -> puro A


def test_seleziona_prior_su_cue_noto():
    rete = ReteFrammento(N=16, T=0.05)
    for c in genera_cue(4, seed=7):
        rete.impara(c)
    ids = sorted(rete.slot.keys())
    rec = rete.slot[ids[0]]
    cue, vis = cue_parziale(rec["Fx"], frazione=0.5, tipo="blocco", seed=1)
    sel = seleziona_prior(rete, cue, vis)
    assert sel["best_id"] == ids[0]
    assert sel["margine"] > 0


def test_salva_report_cl_tmp(tmp_path):
    import os
    righe = sweep_pareto_cl(n_cert=6, n_nuove=8, N=16, T=0.05, seed=7,
                            shift=True)
    assert len(righe) == 8  # ingenua + 3 replay + 4 ewc
    out = salva_report_cl(righe, out_dir=str(tmp_path), N=16, T=0.05,
                          seed=7, shift=True)
    for k in ("csv", "md", "fig"):
        assert os.path.exists(out[k])


def test_salva_report_plast_ood_tmp(tmp_path):
    import os
    plast = misura_plasticita(n_list=[10, 20], N=16, T=0.05, seed=7)
    o1 = salva_report_plasticita(plast, out_dir=str(tmp_path), N=16,
                                 T=0.05, seed=7)
    mix = ood_mix_retrieval(N=16, T=0.05, seed=7, alpha_list=(0.0, 1.0))
    rum = ood_rumore_retrieval(N=16, T=0.05, seed=7, rumori=(0.0, 2.0))
    o2 = salva_report_ood(mix, rum, out_dir=str(tmp_path), N=16, T=0.05,
                          seed=7)
    for o in (o1, o2):
        for k in ("csv", "md", "fig"):
            assert os.path.exists(o[k])
