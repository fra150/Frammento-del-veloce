"""Test del modello 2D (src/frammento_2d.py). Veloci e deterministici."""

import numpy as np

from src.frammento_2d import (
    Param,
    derivata_numerica,
    esperimento_diffusione,
    essenza,
    fedelta,
    griglia,
    input_field,
    invariante_nv,
    laplaciano,
    lfp_sintetico,
    lyapunov,
    metriche,
    riepilogo,
    simula,
    turing_gy,
)


def test_dt_stabile_positivo():
    p = Param(N=32)
    assert p.dt_stabile() > 0
    assert p.dx == p.L / p.N


def test_essenza_massa_normalizzata():
    p = Param(N=32)
    F = essenza(p, massa=1.0)
    assert F.shape == (32, 32)
    assert abs(F.sum() * p.dx ** 2 - 1.0) < 1e-9


def test_laplaciano_costante_nullo():
    p = Param(N=16)
    F = np.ones((p.N, p.N))
    assert np.allclose(laplaciano(F, p.dx), 0.0, atol=1e-12)
    # il laplaciano periodico conserva la media
    rng = np.random.default_rng(0)
    G = rng.random((p.N, p.N))
    assert abs(laplaciano(G, p.dx).sum()) < 1e-9


def test_input_field_normalizzato():
    p = Param(N=32)
    Fin = input_field(p, t=0.05, ampiezza=1.0)
    assert Fin.shape == (32, 32)
    assert abs(Fin.max() - 1.0) < 1e-9


def test_simula_conserva_massa_g0():
    p = Param(N=32, seed=7)
    snap = simula(p, T=0.02, protocollo="stimolo", salva_ogni=5)
    d = snap["diag"]
    # g0 ha R=0 + diffusione conservativa -> massa invariata
    assert abs(d["massa0"][-1] - d["massa0"][0]) < 1e-9
    assert abs(d["massa0"][0] - 1.0) < 1e-9


def test_simula_lyapunov_decresce():
    p = Param(N=32, seed=7)
    snap = simula(p, T=0.02, protocollo="stimolo", salva_ogni=5)
    d = snap["diag"]
    assert d["V"][-1] <= d["V"][0]
    dV = derivata_numerica(d["t"], d["V"])
    assert float(np.mean(dV <= 1e-9)) > 0.5


def test_invariante_nv_struttura_e_stima():
    p = Param(N=64, seed=7)
    for nome, D in (("g0", p.D0), ("gx", p.Dx), ("gy", p.Dy)):
        inv = invariante_nv(p, D, T=0.06)
        assert inv["n"] == 1.0
        assert inv["D_stimato"] > 0
        # stima entro un fattore 2 dal valore vero
        assert 0.5 < inv["D_stimato"] / D < 1.5, nome
    e = esperimento_diffusione(p, p.Dx, T=0.06)
    assert len(e["t"]) == len(e["r2"]) >= 3


def test_metriche_range():
    p = Param(N=32, seed=7)
    snap = simula(p, T=0.02, protocollo="stimolo", salva_ogni=5)
    m = metriche(snap)
    assert np.all(m["qualita"] <= 1.0 + 1e-9)
    assert np.all(m["continuita"] >= -1.0 - 1e-9)
    assert np.all(m["continuita"] <= 1.0 + 1e-9)


def test_fedelta_identica_uguale_uno():
    p = Param(N=32, seed=7)
    snap = simula(p, T=0.02, protocollo="stimolo", salva_ogni=5)
    assert abs(fedelta(snap, snap) - 1.0) < 1e-12


def test_simula_rumore_colorato_rilassamento():
    p = Param(N=16, seed=7)
    snap = simula(p, T=0.03, protocollo="rilassamento", salva_ogni=2,
                  stocastico=True, rumore_bianco=False)
    assert len(snap["t"]) >= 2
    assert np.all(np.isfinite(snap["Fy"][-1]))


def test_lyapunov_nonnegativo():
    p = Param(N=16, seed=7)
    snap = simula(p, T=0.01, protocollo="rilassamento", salva_ogni=2)
    assert np.all(snap["diag"]["V"] >= 0)


def test_lfp_sintetico():
    p = Param(N=32, seed=7)
    snap = simula(p, T=0.02, protocollo="stimolo", salva_ogni=5)
    sig = lfp_sintetico(p, snap["Fy"][-1], fs=1000.0, durata=0.2)
    assert len(sig["lfp"]) == 200
    assert 0.0 <= sig["novita"] <= 1.0


def test_turing_vincolato_a_g0():
    p = Param(N=32, seed=7)
    out = turing_gy(p, passi=400, dt=2.5e-4)
    a, mask = out["a"], out["mask"]
    assert np.all(a >= 0)
    # il pattern vive solo dentro il supporto di g0
    assert np.allclose(a * mask, a)


def test_derivata_numerica_snapshot_singolo():
    # regressione: un solo snapshot non deve far crashare np.gradient
    dV = derivata_numerica(np.array([0.0]), np.array([3.5]))
    assert dV.shape == (1,)
    assert dV[0] == 0.0


def test_riepilogo_snapshot_singolo_non_crash():
    p = Param(N=16, seed=7)
    snap = simula(p, T=0.001, protocollo="stimolo")  # un solo snapshot
    assert "Lyapunov" in riepilogo(snap)


def test_riepilogo_contiene_chiavi():
    p = Param(N=32, seed=7)
    snap = simula(p, T=0.02, protocollo="stimolo", salva_ogni=5)
    txt = riepilogo(snap)
    for chiave in ("massa g0", "novita'", "Lyapunov", "dV/dt"):
        assert chiave in txt


def test_rumore_scaling_sqrt_dt_riduce_fedelta():
    # con Eulero-Maruyama il rumore deve distinguersi: gamma alto -> fedelta' bassa
    from src.studi import valuta as _valuta
    p_base = Param(N=16, seed=7)
    r0 = _valuta(p_base, T=0.03, salva_ogni=2, T_rec=0.03)
    p_hi = Param(N=16, gamma=0.25, seed=7)
    r1 = _valuta(p_hi, T=0.03, salva_ogni=2, T_rec=0.03)
    assert r1["fedelta"] < r0["fedelta"] - 0.005, (r0["fedelta"], r1["fedelta"])
    assert r1["fedelta"] < 1.0


def test_fy_nonnegativa_con_rumore_forte():
    p = Param(N=16, gamma=0.25, seed=7)
    snap = simula(p, T=0.03, protocollo="stimolo", salva_ogni=2)
    assert np.all(np.asarray(snap["Fy"][-1]) >= 0.0)


def test_invariante_lambda_g_presente_e_finito():
    from src.frammento_2d import verifica_invarianza as _ver
    p = Param(N=64, seed=7)
    for D in (p.D0, p.Dx, p.Dy):
        inv = invariante_nv(p, D, T=0.06)
        assert np.isfinite(inv["lambda_g"]) and inv["lambda_g"] > 0
        assert np.isfinite(inv["eta_g"]) and inv["eta_g"] > 0
    assert _ver(p, T=0.06)["cv"] < 0.2
