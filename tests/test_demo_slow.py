"""Test slow: pipeline demo completa (stessi parametri di src/demo_figure.py:main).

Eseguiti solo con:  pytest tests/ -q --run-slow
Skippati di default (vedi tests/conftest.py).
"""

import os

import pytest

pytestmark = pytest.mark.slow

import src.demo_figure as demo
from src.frammento_1d import FrammentoDelVeloce, Params, plot
from src.frammento_2d import Param, simula


FIGURE_ATTESE = [
    "fig01_tre_livelli.png",
    "fig02_evoluzione.png",
    "fig03_lyapunov_massa.png",
    "fig04_metriche.png",
    "fig05_turing_gy.png",
    "fig06_invariante_nv.png",
    "fig07_lfp.png",
]


def test_demo_completa_7_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "OUT", str(tmp_path))
    p = Param(N=80, D0=0.008, Dx=0.012, Dy=0.003,
              alpha=4.0, kappa_x=0.8, beta=1.4, K=1.15,
              gamma=0.015, kappa_y=0.18, seed=7)
    snap = simula(p, T=0.22, protocollo="stimolo", salva_ogni=40, stocastico=True)
    d = snap["diag"]
    assert abs(d["massa0"][-1] - d["massa0"][0]) < 1e-9  # conservazione g0
    assert d["V"][-1] <= d["V"][0]                       # Lyapunov decresce

    demo.fig_tre_livelli(snap)
    demo.fig_evoluzione(snap)
    demo.fig_diagnostica(snap)
    demo.fig_metriche(snap)
    demo.fig_turing(p)
    demo.fig_invariante(p)
    demo.fig_lfp(snap)

    for nome in FIGURE_ATTESE:
        path = os.path.join(str(tmp_path), nome)
        assert os.path.isfile(path), nome
        assert os.path.getsize(path) > 0, nome


def test_demo_main(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "OUT", str(tmp_path))
    demo.main()
    for nome in FIGURE_ATTESE:
        path = os.path.join(str(tmp_path), nome)
        assert os.path.isfile(path), nome
        assert os.path.getsize(path) > 0, nome


def test_modello_1d_completo(tmp_path):
    p = Params(T=2.0, N=256, seed=0)
    sim = FrammentoDelVeloce(p)
    sim.run(verbose=False)
    drift = abs(sim.hist.mass[-1] - sim.M0) / sim.M0
    assert drift < 1e-9
    plot(sim, os.path.join(str(tmp_path), "frammento_1d.png"))
    assert os.path.isfile(os.path.join(str(tmp_path), "frammento_1d.png"))
