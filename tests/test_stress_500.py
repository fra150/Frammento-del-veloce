"""Stress test rapido (sottoinsieme): verifica la catena g0->gf su poche domande."""

from src.stress_500 import genera_domande, interroga
from src.frammento_gf import MemoriaGF


def test_genera_domande_con_repliche():
    doms = genera_domande(n=50, seed=7)
    assert len(doms) == 50
    reps = [d for d in doms if d["ripete"] is not None]
    assert len(reps) == 1 and reps[0]["ripete"] == 0
    # la replica condivide i parametri dell'originale
    assert reps[0]["seed_domanda"] == doms[0]["seed_domanda"]


def test_interroga_catena_g0_gf():
    mem = MemoriaGF()
    doms = genera_domande(n=6, seed=7)
    for d in doms:
        r = interroga(d, N=16, T=0.05, mem=mem)
        assert set(r) >= {"qualita", "novita_rel", "quiete", "cert", "hit", "massa_g0_drift"}
        assert 0.0 <= r["novita_rel"] <= 2.0
        assert r["massa_g0_drift"] < 1e-6


def test_replica_da_cache_hit():
    mem = MemoriaGF()
    doms = genera_domande(n=26, seed=7)
    for d in doms[:25]:
        interroga(d, N=16, T=0.05, mem=mem)
    r = interroga(doms[25], N=16, T=0.05, mem=mem)  # replica di doms[0] se certificata
    # se l'originale era certificato, la replica deve dare hit
    r0 = interroga(doms[0], N=16, T=0.05, mem=MemoriaGF())
    if r0["cert"]:
        assert r["hit"] is True
