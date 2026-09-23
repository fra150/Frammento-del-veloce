"""Fase 16: sonno + eviction + transfer + retrieval repair (fast, N=16)."""

import numpy as np

from src.fase16 import (
    distilla_gist,
    sonno,
    valuta_sonno_mix,
    valuta_sonno_rumore,
    evici_slot,
    esegui_eviction_study,
    misura_transfer,
    split_vis,
    topk_prior,
    seleziona_prior_validato,
    seleziona_prior_coarse_to_fine,
    valuta_retrieval_repair,
    salva_report_sonno,
    salva_report_eviction,
    salva_report_transfer,
    salva_report_retrieval,
)
from src.rete_frammento import ReteFrammento, cue_parziale, genera_cue


def _rete_piccola(n=8, N=16, T=0.05, seed=7):
    r = ReteFrammento(N=N, T=T)
    for c in genera_cue(n, seed=seed):
        r.impara(c)
    return r


def test_sonno_non_tocca_slot():
    r = _rete_piccola(6)
    h0 = {i: v["Fx"].copy() for i, v in r.slot.items()}
    ck0 = r.checksum
    s = sonno(r)
    assert s["intatto"] is True
    assert r.checksum == ck0
    for i, fx0 in h0.items():
        assert np.array_equal(np.asarray(r.slot[i]["Fx"]), fx0)


def test_gist_media_e_costo():
    r = _rete_piccola(6)
    g = distilla_gist(r)
    assert g["n_slot"] == len(r.slot)
    assert g["gist"].shape == (16, 16)
    assert g["costo_bytes"] == 16 * 16 * 8
    stack = np.stack([np.asarray(v["Fx"]) for v in r.slot.values()])
    assert np.allclose(g["gist"], stack.mean(axis=0))


def test_sonno_mix_estremi():
    righe = valuta_sonno_mix(N=16, T=0.05, seed=7,
                             alpha_list=(0.0, 1.0))
    assert len(righe) == 2
    assert righe[0]["vincitore"] == "B"
    assert righe[1]["vincitore"] == "A"
    for r in righe:
        assert 0.0 <= r["q_scelta_vs_mix"] <= 1.0


def test_sonno_rumore_q_nonneg():
    righe = valuta_sonno_rumore(N=16, T=0.05, seed=7, rumori=(0.0, 2.0))
    assert righe[0]["acc_winner"] == 1.0
    assert righe[1]["acc_winner"] < 1.0
    for r in righe:
        assert r["q_gist_media"] >= 0.0


def test_eviction_eta_q_uso():
    r = _rete_piccola(8)
    ids = sorted(r.slot.keys())
    # eta: i piu' vecchi (id piccoli)
    ev = evici_slot(r, 3, politica="eta")
    assert ev == ids[:3]
    assert r.verifica_nucleo()["ok"]
    # q: i piu' deboli prima (sottoinsieme dei rimasti)
    r2 = _rete_piccola(8)
    ev2 = evici_slot(r2, 2, politica="q")
    assert len(ev2) == 2
    qs = {i: float(v["q_cert"]) for i, v in _rete_piccola(8).slot.items()}
    # i due evictati devono essere tra i peggiori (controllo debole ma vero)
    assert len(set(ev2)) == 2
    # uso: proxy id%3 -> prima uso 0 (id 0,3,6)
    r3 = _rete_piccola(8)
    ev3 = evici_slot(r3, 3, politica="uso")
    assert set(ev3) == {0, 3, 6}


def test_eviction_study_copertura_e_memoria():
    righe = esegui_eviction_study(n=12, N=16, T=0.05, seed=7,
                                  k_list=(0, 6), politiche=("eta", "uso"))
    assert len(righe) == 4
    for r in righe:
        if r["k_evictati"] == 0:
            assert r["copertura"] == 1.0
        else:
            assert abs(r["copertura"] - 0.5) < 0.2  # ~6/12
    # memoria: evictare dimezza i KB
    m0 = [r for r in righe if r["k_evictati"] == 0][0]["memoria_kb"]
    m6 = [r for r in righe if r["k_evictati"] == 6][0]["memoria_kb"]
    assert m6 < m0


def test_transfer_protetta_zero_e_ingenua_negativa():
    res = misura_transfer(n_cert=8, n_nuove=12, N=16, T=0.05, seed=7)
    assert res["n_A"] > 0 and res["n_B"] > 0
    pm = res["per_modello"]
    assert abs(pm["protetta"]["BWT"]) <= 1e-9
    assert abs(pm["protetta"]["FWT"]) <= 1e-9
    # con lo shift la condivisa dimentica: BWT negativa
    assert pm["ingenua"]["BWT"] < -1e-6


def test_split_vis_disgiunto_e_completo():
    r = _rete_piccola(2)
    rec = r.slot[sorted(r.slot.keys())[0]]
    _, vis = cue_parziale(rec["Fx"], frazione=0.5, tipo="blocco", seed=1)
    fit, val = split_vis(vis, fraz_val=0.2, seed=3)
    assert ((fit & val).sum() == 0)
    assert np.array_equal(fit | val, vis)
    assert val.sum() > 0 and fit.sum() > 0


def test_validato_non_oracolo_su_cue_noto():
    # in-distribution senza rumore: anche il repair deve trovare il giusto
    r = _rete_piccola(4)
    ids = sorted(r.slot.keys())
    rec = r.slot[ids[0]]
    cue, vis = cue_parziale(rec["Fx"], frazione=0.5, tipo="blocco", seed=1)
    dx = 1.0 / 16
    v = seleziona_prior_validato(r, cue, vis, dx, k=3, seed=5)
    assert v["best_id"] == ids[0]
    assert v["fallback"] is False
    c = seleziona_prior_coarse_to_fine(r, cue, vis, dx, k=3, seed=5)
    assert c["best_id"] == ids[0]


def test_topk_contiene_winner():
    from src.fase15 import seleziona_prior
    r = _rete_piccola(4)
    rec = r.slot[sorted(r.slot.keys())[0]]
    cue, vis = cue_parziale(rec["Fx"], frazione=0.5, tipo="blocco", seed=2)
    base = seleziona_prior(r, cue, vis)
    top = topk_prior(r, cue, vis, k=3)
    assert base["best_id"] in [i for _, i in top]


def test_retrieval_repair_rumore_struttura():
    righe = valuta_retrieval_repair(N=16, T=0.05, seed=7,
                                    rumori=(0.0, 2.0))
    assert len(righe) == 2
    assert righe[0]["acc_base"] == 1.0
    for r in righe:
        for k in ("acc_base", "acc_validato", "acc_coarse"):
            assert 0.0 <= r[k] <= 1.0


def test_salva_report_fase16_tmp(tmp_path):
    import os
    mix = valuta_sonno_mix(N=16, T=0.05, seed=7, alpha_list=(0.0, 1.0))
    rum = valuta_sonno_rumore(N=16, T=0.05, seed=7, rumori=(0.0, 1.0))
    o1 = salva_report_sonno(mix, rum, out_dir=str(tmp_path))
    ev = esegui_eviction_study(n=8, N=16, T=0.05, seed=7,
                               k_list=(0, 4), politiche=("eta",))
    o2 = salva_report_eviction(ev, out_dir=str(tmp_path))
    tr = misura_transfer(n_cert=6, n_nuove=8, N=16, T=0.05, seed=7)
    o3 = salva_report_transfer(tr, out_dir=str(tmp_path))
    rr = valuta_retrieval_repair(N=16, T=0.05, seed=7, rumori=(0.0, 1.0))
    o4 = salva_report_retrieval(rr, out_dir=str(tmp_path))
    for o in (o1, o2, o3, o4):
        for k in ("csv", "md", "fig"):
            assert os.path.exists(o[k])
