"""Test per il futuro modulo src/frammento_gf.py (quiete / certificazione / cache).

Contratto API ipotizzato (il coder parallelo deve implementarlo cosi'):
- verifica_quiete(Fo, Fx, essenza, dx, eps=0.05, delta=0.05,
                  Fy=None, V_hist=None, budget_novita_rel=0.25) -> dict con 'attivo'
- certifica_frammento(Fo, Fx, Fy, essenza, dx, diag=None, Fin=None,
                      eps=0.05, delta=0.05,
                      soglia_qualita=0.60, budget_novita_rel=0.30)
      -> dict con 'certificato', 'qualita'
- correggi_micro_errori(Fx, Fo, fattore=0.1) -> np.ndarray (stessa shape)
- class MemoriaGF():
      chiave(arr: np.ndarray) -> hashable (str)
      salva(chiave, payload) -> None
      richiama(chiave) -> (hit: bool, payload)
      stats() -> dict con 'hits'/'misses'

Stile: N=16 per velocita', nessuna figura, nessun file. Totali <2s.
"""

import numpy as np
import pytest

from src.frammento_2d import Param, essenza, simula
from src.frammento_gf import (
    MemoriaGF,
    certifica_frammento,
    correggi_micro_errori,
    verifica_quiete,
)


def test_quiete_attiva_su_essenza():
    p = Param(N=16)
    ess = essenza(p)
    Fo = ess.copy()
    Fx = ess.copy()
    out = verifica_quiete(Fo, Fx, ess, p.dx)
    assert isinstance(out, dict)
    assert "attivo" in out
    assert bool(out["attivo"]) is True


def test_quiete_spenta_se_perturbato():
    p = Param(N=16)
    ess = essenza(p)
    Fo = ess.copy()
    Fx = ess.copy() + 5.0  # bump grande uniforme: fuori soglia eps/delta
    out = verifica_quiete(Fo, Fx, ess, p.dx)
    assert isinstance(out, dict)
    assert "attivo" in out
    assert bool(out["attivo"]) is False


def test_quiete_spenta_se_Fy_esplosa():
    p = Param(N=16)
    ess = essenza(p)
    Fo = ess.copy()
    Fx = ess.copy()
    Fy = np.full_like(ess, 5.0)  # novita' esplosa: oltre budget_novita_rel
    out = verifica_quiete(Fo, Fx, ess, p.dx, Fy=Fy)
    assert isinstance(out, dict)
    assert "attivo" in out
    assert bool(out["attivo"]) is False


def test_certifica_base_piccola_simulazione():
    p = Param(N=16, seed=7)
    snap = simula(p, T=0.03, stocastico=False, salva_ogni=2)
    Fo = np.asarray(snap["F0"][-1])
    Fx = np.asarray(snap["Fx"][-1])
    Fy = np.asarray(snap["Fy"][-1])
    ess = np.asarray(snap["essenza"])
    out = certifica_frammento(Fo, Fx, Fy, ess, p.dx)
    assert isinstance(out, dict)
    assert "certificato" in out
    assert "qualita" in out
    assert np.isfinite(float(out["qualita"]))


def test_non_certifica_se_non_quiete():
    # anti-tautologia: campo corrotto -> mai certificato True
    p = Param(N=16, seed=7)
    snap = simula(p, T=0.03, stocastico=False, salva_ogni=2)
    Fo = np.asarray(snap["F0"][-1])
    Fx = np.asarray(snap["Fx"][-1])
    Fy = np.asarray(snap["Fy"][-1])
    ess = np.asarray(snap["essenza"])
    Fx_bad = Fx + 5.0
    out = certifica_frammento(Fo, Fx_bad, Fy, ess, p.dx)
    assert isinstance(out, dict)
    assert "certificato" in out
    assert bool(out["certificato"]) is False


def test_cache_hit_costo_zero():
    p = Param(N=16)
    ess = essenza(p)
    mem = MemoriaGF()
    k = mem.chiave(ess)
    # chiave deterministica per stesso contenuto
    assert mem.chiave(ess.copy()) == k
    payload = {"qualita": 0.99, "tag": "prova"}
    mem.salva(k, payload)
    hit, got = mem.richiama(k)
    assert bool(hit) is True
    assert got == payload
    hit2, got2 = mem.richiama(k)
    assert bool(hit2) is True
    assert got2 == payload
    s = mem.stats()
    assert isinstance(s, dict)
    if "hits" in s:
        assert int(s["hits"]) >= 1
    elif "hit" in s:
        assert int(s["hit"]) >= 1
    else:  # fallback tollerante su nome chiave
        numerici = [v for v in s.values() if isinstance(v, (int, float))]
        assert any(v >= 1 for v in numerici)


def test_correzione_non_autocertifica():
    # la correzione avvicina a Fo ma NON restituisce un certificato:
    # certifica_frammento va richiamata esplicitamente a parte
    p = Param(N=16)
    ess = essenza(p)
    Fo = ess.copy()
    rng = np.random.default_rng(0)
    Fx = Fo + 0.05 * rng.standard_normal((p.N, p.N))  # micro-errore
    Fx_corr = correggi_micro_errori(Fx, Fo, fattore=0.1)
    assert isinstance(Fx_corr, np.ndarray)
    assert not isinstance(Fx_corr, dict)
    assert np.shape(Fx_corr) == np.shape(Fo)
    d_prima = float(np.linalg.norm(Fx - Fo))
    d_dopo = float(np.linalg.norm(np.asarray(Fx_corr) - Fo))
    assert d_dopo < d_prima
