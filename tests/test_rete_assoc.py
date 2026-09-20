"""Test del richiamo associativo (rete_frammento: cue parziali, ricostruzione).

Fast (<15 s, N=16): cue parziale, ricostruzione deterministica e migliore
del riempimento semplice, invarianza della protetta, accoppiamento della
condivisa al set, smoke del runner.
Slow (full 200+800, N=32): shift di classe, degrado condivisa, protetta
bit-identica.
"""

import numpy as np
import pytest

from src.rete_frammento import (
    ReteFrammento, ReteIngenuaCondivisa, genera_cue, cue_parziale,
    ricostruisci_associativo, esegui_test_associativo, _q_regione)


def test_cue_parziale_blocco():
    p = np.arange(16 * 16, dtype=float).reshape(16, 16)
    cue, vis = cue_parziale(p, frazione=0.25, tipo="blocco", seed=0)
    assert vis.shape == p.shape
    frac = float((~vis).mean())
    assert 0.15 < frac < 0.35
    # la parte mancante e' azzerata
    assert np.allclose(cue[~vis], 0.0)
    # i visibili restano intatti (senza rumore)
    assert np.allclose(cue[vis], p[vis])
    # deterministico
    cue2, vis2 = cue_parziale(p, frazione=0.25, tipo="blocco", seed=0)
    assert np.array_equal(vis, vis2) and np.allclose(cue, cue2)


def test_cue_parziale_casuale_e_rumore():
    rng = np.random.default_rng(3)
    p = rng.standard_normal((16, 16))
    cue, vis = cue_parziale(p, frazione=0.5, tipo="casuale", seed=1)
    frac = float((~vis).mean())
    assert 0.35 < frac < 0.65
    cue_n, _ = cue_parziale(p, frazione=0.5, tipo="casuale", rumore=0.1, seed=1)
    # il rumore cambia i valori (almeno qualche pixel)
    assert not np.allclose(cue, cue_n)


def test_ricostruzione_deterministica():
    rng = np.random.default_rng(0)
    prior = rng.standard_normal((16, 16))
    Fx = prior + 0.5 * rng.standard_normal((16, 16))
    cue, vis = cue_parziale(Fx, frazione=0.5, tipo="blocco", seed=2)
    r1 = ricostruisci_associativo(cue, vis, prior, 1.0 / 16, T_rec=2.0)
    r2 = ricostruisci_associativo(cue, vis, prior, 1.0 / 16, T_rec=2.0)
    assert np.array_equal(r1, r2)
    # ancoraggio: i pixel visibili restano quelli del cue
    assert np.allclose(r1[vis], cue[vis])


def test_ricostruzione_meglio_del_riempimento():
    # su una memoria reale: la ricostruzione batte il riempimento con la media
    rete = ReteFrammento(N=16, T=0.05)
    cue_d = genera_cue(3, seed=11)
    for c in cue_d:
        rete.impara(c)
    cid = sorted(rete.slot.keys())[0]
    rec = rete.slot[cid]
    cue, vis = cue_parziale(rec["Fx"], frazione=0.5, tipo="blocco", seed=5)
    Fx_rec = ricostruisci_associativo(cue, vis, rec["Fo"], 1.0 / 16,
                                      T_rec=3.0)
    q_rec = _q_regione(Fx_rec, rec["Fx"], ~vis)
    media = float(cue[vis].mean())
    Fx_med = cue.copy()
    Fx_med[~vis] = media
    q_med = _q_regione(Fx_med, rec["Fx"], ~vis)
    assert q_rec > q_med + 0.1


def test_protetta_invariante_dopo_nuovi_apprendimenti():
    # la protetta non cambia richiamo dopo nuovi apprendimenti (frozen)
    cueA = genera_cue(8, seed=7, amp_range=(3.0, 8.0), regione="sinistra")
    cueB = genera_cue(12, seed=77, amp_range=(3.0, 8.0), regione="destra")
    for k, c in enumerate(cueB):
        c["id"] = 100 + k
    rete = ReteFrammento(N=16, T=0.05)
    for c in cueA:
        rete.impara(c)
    ids_A = sorted(rete.slot.keys())
    prima = {cid: rete.richiama_associativo(cid, 0.5, seed_cue=cid)["q_mask_rec"]
             for cid in ids_A}
    for c in cueB:
        rete.impara(c)
    for cid in ids_A:
        dopo = rete.richiama_associativo(cid, 0.5, seed_cue=cid)["q_mask_rec"]
        assert dopo == prima[cid]  # bit-identico


def test_condivisa_accoppiata_al_set():
    # la condivisa cambia richiamo dei vecchi ricordi dopo lo shift
    cueA = genera_cue(8, seed=7, amp_range=(3.0, 8.0), regione="sinistra")
    cueB = genera_cue(12, seed=77, amp_range=(3.0, 8.0), regione="destra")
    for k, c in enumerate(cueB):
        c["id"] = 100 + k
    ing = ReteIngenuaCondivisa(N=16, T=0.05)
    for c in cueA:
        ing.impara(c)
    P_A = ing.P.copy()
    for c in cueB:
        ing.impara(c)
    ids_A = sorted(ing.record.keys())
    differenze = []
    for cid in ids_A:
        r_ctrl = ing.richiama_associativo(cid, 0.5, seed_cue=cid, P_rif=P_A)
        r_now = ing.richiama_associativo(cid, 0.5, seed_cue=cid)
        differenze.append(abs(r_now["q_mask_rec"] - r_ctrl["q_mask_rec"]))
    # accoppiata: almeno un ricordo cambia richiamo dopo lo shift
    assert max(differenze) > 1e-6


def test_esegui_assoc_smoke():
    res = esegui_test_associativo(n_cert=8, n_nuove=12, N=16, T=0.05,
                                  seed=7, frazioni=(0.25, 0.50))
    assert res["n_cert"] > 0
    assert res["nucleo_ok"] is True
    assert res["verifica_protetta_max_diff"] == 0.0
    assert res["shift_rel"] > 0.0
    for fq in (0.25, 0.50):
        a = res["agg"][fq]
        assert set(a) >= {"q_mask_rec", "q_mask_media", "q_mask_core",
                          "q_mask_ing", "q_mask_ing0", "degrado_ing"}
        assert a["q_mask_rec"] > a["q_mask_media"]


@pytest.mark.slow
def test_assoc_full_shift_slow():
    res = esegui_test_associativo(n_cert=200, n_nuove=800, N=32, T=0.10,
                                  seed=7)
    assert res["n_cert"] >= 150
    assert res["nucleo_ok"] is True
    # la protetta e' bit-identica prima/dopo lo shift
    assert res["verifica_protetta_max_diff"] == 0.0
    # lo shift c'e' ed e' misurabile
    assert res["shift_rel"] > 0.05
    # degrado della condivisa: cresce col buco, massimo a f=0.75
    a75 = res["agg"][0.75]
    assert a75["degrado_ing"] > 0.01
    assert a75["degrado_ing_max"] > a75["degrado_ing"]
    # la ricostruzione protetta batte i riempimenti semplici
    a25 = res["agg"][0.25]
    assert a25["q_mask_rec"] > a25["q_mask_media"] + 0.3
    assert a25["q_mask_rec"] > a25["q_mask_arm"] + 0.1
