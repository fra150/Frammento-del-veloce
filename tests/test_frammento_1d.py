"""Test del simulatore 1D (src/frammento_1d.py). Veloci e deterministici."""

import numpy as np
import pytest

from src.frammento_1d import Domain, FrammentoDelVeloce, Params


def test_params_dt_automatico_e_cfl():
    p = Params(N=64)
    assert p.dt is not None and p.dt > 0


def test_domain_laplaciano_costante_periodico():
    p = Params(N=64, periodic=True)
    dom = Domain(p)
    assert np.allclose(dom.laplacian(np.ones(p.N)), 0.0, atol=1e-10)
    assert abs(dom.integral(np.ones(p.N)) - p.L) < 1e-9


def test_domain_neumann_ok():
    p = Params(N=64, periodic=False)
    dom = Domain(p)
    assert dom.laplacian(np.ones(p.N)).shape == (p.N,)
    assert dom.l2(np.ones(p.N)) > 0


def test_run_conserva_massa():
    p = Params(N=64, T=0.2, seed=0, conserve_mass=True, save_every=1)
    sim = FrammentoDelVeloce(p)
    sim.run(verbose=False)
    drift = abs(sim.hist.mass[-1] - sim.M0) / sim.M0
    assert drift < 1e-9
    assert len(sim.hist.t) > 2


def test_quality_e_continuity_range():
    p = Params(N=64, T=0.1, seed=0)
    sim = FrammentoDelVeloce(p)
    sim.run(verbose=False)
    assert 0.0 <= sim.quality() <= 1.0
    assert all(0.0 <= q <= 1.0 for q in sim.hist.quality)
    assert all(0.0 <= c <= 1.0 for c in sim.hist.continuity)


def test_fidelity_test_passa():
    p = Params(N=64, T=0.1, seed=0)
    sim = FrammentoDelVeloce(p)
    sim.run(verbose=False)
    assert sim.fidelity_test() > 0.5


def test_project_novelty_rispetta_budget():
    p = Params(N=64, T=0.05, seed=0, novelty_budget=0.1)
    sim = FrammentoDelVeloce(p)
    F = sim.g0.F_ref + 10.0 * np.ones(p.N)
    Fp = sim.project_novelty(F)
    assert sim.dom.l2(Fp - sim.g0.F_ref) <= 0.1 * sim.dom.l2(sim.g0.F_ref) + 1e-9


def test_report_non_crash_e_valori_sani(capsys):
    p = Params(N=64, T=0.05, seed=0)
    sim = FrammentoDelVeloce(p)
    sim.run(verbose=False)
    n_steps = int(p.T / p.dt)
    sim.report(n_steps)  # stampa il riepilogo, non deve sollevare eccezioni
    out = capsys.readouterr().out
    for chiave in ("massa iniziale M0", "deriva di massa", "V finale", "fedelt"):
        assert chiave in out
    assert sim.hist.V[-1] >= 0
