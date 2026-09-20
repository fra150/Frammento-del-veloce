"""Rete che non distrugge: nucleo frozen, gf unico canale, test 1000.

Fast (<5 s, N=16): nucleo checksum, solo-gf-scrive, capacita',
interferenza zero su piccolo, ingenua che degrada, cache hit.
Slow (full 200+800, N=32): test dei 1000 vero, Q piatta.
"""

import numpy as np
import pytest

from src.rete_frammento import (
    ReteFrammento, ReteIngenuaCondivisa, esegui_test_1000, genera_cue)


def test_genera_cue_indipendenti():
    c = genera_cue(30, seed=7)
    assert len(c) == 30
    assert len({x["seed_domanda"] for x in c}) > 25  # quasi tutte distinte


def test_nucleo_immutabile_checksum():
    r = ReteFrammento(N=16, T=0.05)
    ck0 = r.checksum
    for cue in genera_cue(5, seed=7):
        r.impara(cue)
    v = r.verifica_nucleo()
    assert v["ok"] is True
    assert v["checksum"] == ck0
    # property restituisce copia: mutarla non corrompe il nucleo
    copia = r.nucleo
    copia += 99.0
    assert r.verifica_nucleo()["ok"] is True


def test_solo_gf_scrive():
    r = ReteFrammento(N=16, T=0.05)
    n0 = len(r.slot)
    # scrittura diretta senza certificato: rifiutata
    ok = r._scrivi_solo_se_certificato(999, {"finto": 1}, False)
    assert ok is False
    assert len(r.slot) == n0
    # impara scrive solo se certificato
    scritti = 0
    for cue in genera_cue(5, seed=11):
        out = r.impara(cue)
        if out["cert"]:
            assert out["scritto"] is True
            scritti += 1
        else:
            assert out["scritto"] is False
    assert len(r.slot) == scritti


def test_capacita_rifiuta_vs_espandi():
    cue = genera_cue(6, seed=5)
    r = ReteFrammento(N=16, T=0.05, capacita_max=2, politica="rifiuta")
    for c in cue:
        r.impara(c)
    assert len(r.slot) <= 2  # mai oltre, mai overwrite
    r2 = ReteFrammento(N=16, T=0.05, capacita_max=2, politica="espandi")
    for c in cue:
        r2.impara(c)
    # espandi: accoglie tutti i certificati tra le 6 cue
    n_cert = sum(1 for c in cue
                 for _ in [r2.cue_note.get(c["id"])])
    assert len(r2.slot) >= len(r.slot)


def test_interferenza_zero_piccolo():
    res = esegui_test_1000(n_cert=10, n_nuove=20, N=16, T=0.05, seed=7)
    assert res["n_certificati"] > 0
    assert res["nucleo_ok"] is True
    assert res["max_degrado"] <= 1e-9
    assert res["distrutti"] == 0
    assert res["successo"] is True


def test_ingenua_degrada():
    # la baseline condivisa DEVE degradare: se non degrada il test e' vacuo
    res = esegui_test_1000(n_cert=10, n_nuove=20, N=16, T=0.05, seed=7)
    assert res["max_degrado_ing"] > 1e-6


def test_richiamo_cache_hit():
    r = ReteFrammento(N=16, T=0.05)
    cue = genera_cue(3, seed=7)
    for c in cue:
        r.impara(c)
    for cid in list(r.slot.keys()):
        rr = r.richiama_ricalcola(cid)
        assert rr["hit"] is True
        assert rr["degrado"] <= 1e-9


@pytest.mark.slow
def test_1000_completo_lento():
    res = esegui_test_1000(n_cert=200, n_nuove=800, N=32, T=0.10, seed=7)
    assert res["n_certificati"] >= 150  # regime sano: quasi tutte certificate
    assert res["nucleo_ok"] is True
    assert res["max_degrado"] <= 1e-9
    assert res["distrutti"] == 0
    assert res["successo"] is True
