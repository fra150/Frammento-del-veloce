"""Fase 16-bis: retrieval v2 (pesi, niente blur) + transfer v2 (fast)."""

import numpy as np

from src.fase16 import (
    campo_vero_cue,
    essenza_campo,
    fedelta_condivisa,
    fedelta_protetta,
    misura_transfer_v2,
    pesi_varianza,
    salva_report_retrieval_v2,
    salva_report_transfer_v2,
    seleziona_prior_v2,
    sonno,
    valuta_retrieval_v2,
)
from src.fase15 import ReteIngenuaCondivisa
from src.rete_frammento import (
    ReteFrammento,
    cue_parziale,
    genera_cue,
)


def _rete(n=6, N=16, T=0.05, seed=7):
    r = ReteFrammento(N=N, T=T)
    for c in genera_cue(n, seed=seed):
        r.impara(c)
    return r


def test_essenza_campo_forma_e_determinismo():
    e1 = essenza_campo(16)
    e2 = essenza_campo(16)
    assert e1.shape == (16, 16)
    assert np.array_equal(e1, e2)


def test_pesi_media_uno_e_nonnegativi():
    r = _rete(4)
    rec = r.slot[sorted(r.slot.keys())[0]]
    _, vis = cue_parziale(rec["Fx"], frazione=0.5, tipo="blocco", seed=1)
    w, fb = pesi_varianza(r, vis)
    assert fb is False
    assert (w >= 0.0).all()
    assert abs(float(w[vis].mean()) - 1.0) < 1e-9


def test_v2_esatto_sul_pulito_guardia_97():
    # a rumore zero il v2 DEVE trovare il giusto (score 0 per costruzione)
    r = _rete(4)
    for cid in sorted(r.slot.keys()):
        rec = r.slot[cid]
        cue, vis = cue_parziale(rec["Fx"], frazione=0.5, tipo="blocco",
                                rumore=0.0, seed=cid * 13 + 1)
        v = seleziona_prior_v2(r, cue, vis)
        assert v["best_id"] == cid
        assert v["best_mse"] == 0.0
        assert v["fallback_pesi"] is False


def test_v2_qualita_non_crolla_sul_pulito():
    righe = valuta_retrieval_v2(N=16, T=0.05, seed=7, rumori=(0.0,))
    assert righe[0]["acc_base"] == 1.0
    assert righe[0]["acc_v2"] == 1.0  # doppio criterio, prima meta'
    assert righe[0]["q_v2_media"] >= righe[0]["q_base_media"] - 1e-9


def test_campo_vero_deterministico():
    c = genera_cue(1, seed=7)[0]
    f1, _ = campo_vero_cue(c, 16, 0.05)
    f2, _ = campo_vero_cue(c, 16, 0.05)
    assert np.array_equal(f1, f2)


def test_fedelta_protetta_uno_condivisa_scende():
    r = _rete(4)
    for cid in sorted(r.slot.keys()):
        fp = fedelta_protetta(r, cid)
        assert fp["F"] == 1.0
        assert fp["max_abs_diff"] == 0.0
    # condivisa dopo shift: la deriva c'e' (fedelta' < 1)
    ing = ReteIngenuaCondivisa(N=16, T=0.05)
    A = genera_cue(6, seed=7, amp_range=(3.0, 8.0), regione="sinistra")
    B = genera_cue(8, seed=107, amp_range=(3.0, 8.0), regione="destra")
    for k, c in enumerate(B):
        c["id"] = 100 + k
    for c in A + B:
        ing.impara(c)
    f0 = [fedelta_condivisa(ing, c["id"]) for c in A]
    assert min(f0) < 1.0  # la deriva ha mosso i campi


def test_transfer_v2_separa_su_fedelta():
    res = misura_transfer_v2(n_cert=8, n_nuove=12, N=16, T=0.05, seed=7)
    pm = res["per_modello"]
    assert abs(pm["protetta"]["BWT_fid"]) <= 1e-9
    assert pm["ingenua"]["BWT_fid"] < -1e-6  # il forgetting che la Q nasconde
    for m, v in pm.items():
        assert 0.0 <= v["FWT_zero"] <= 1.0
        assert -1.0 <= v["BWT_fid"] <= 1.0


def test_reports_bis_tmp(tmp_path):
    import os
    rr = valuta_retrieval_v2(N=16, T=0.05, seed=7, rumori=(0.0, 1.0))
    o1 = salva_report_retrieval_v2(rr, out_dir=str(tmp_path))
    tr = misura_transfer_v2(n_cert=6, n_nuove=8, N=16, T=0.05, seed=7)
    o2 = salva_report_transfer_v2(tr, out_dir=str(tmp_path))
    for o in (o1, o2):
        for k in ("csv", "md", "fig"):
            assert os.path.exists(o[k])
    assert o1["verdetto"] in ("PROMOSSO", "BOCCIATO")
