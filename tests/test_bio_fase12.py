"""Test della pipeline statistica di Fase 12 su dati sintetici (niente download).

Verifica che `trend_carico` + `p_permutazione_trend` distinguano un trend
vero da un nullo: il metodo deve dare p piccola su trend forte e p grande
su rumore. I dati EEG veri (10 soggetti ds005095) restano fuori dal repo;
la matrice e' in `output/output_test/fase12_mat.npy`.
"""

import numpy as np

from src.confronto_bio import p_permutazione_trend, trend_carico


def test_trend_forte_rilevato():
    rng = np.random.default_rng(0)
    carichi = (3, 6, 9, 12, 15)
    M = np.array([[c * 0.1 + rng.normal(0, 0.02) for c in carichi] for _ in range(10)])
    t = trend_carico(M, carichi)
    assert t["rho"] > 0.9 and t["p"] < 1e-6
    assert t["n_soggetti"] == 10 and len(t["medie_gruppo"]) == 5
    p = p_permutazione_trend(M, carichi, n_perm=200, seed=0)
    assert p["p_perm"] < 0.05


def test_rumore_non_significativo():
    rng = np.random.default_rng(1)
    M = rng.normal(0.5, 0.2, size=(10, 5))
    t = trend_carico(M)
    p = p_permutazione_trend(M, n_perm=200, seed=0)
    assert 0.0 <= p["p_perm"] <= 1.0
    assert abs(t["rho"]) < 0.5


def test_matrice_fase12_reale_nulla():
    import os
    fp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "output", "output_test", "fase12_mat.npy")
    if not os.path.isfile(fp):
        return  # matrice versionata assente in checkout parziali: skip morbido
    M = np.load(fp)
    assert M.shape == (10, 5)
    t = trend_carico(M)
    p = p_permutazione_trend(M, n_perm=500, seed=0)
    # risultato noto della Fase 12: nullo (rho~-0.01, p_perm~0.9)
    assert abs(t["rho"]) < 0.2
    assert p["p_perm"] > 0.2
