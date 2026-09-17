#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frammento del veloce - implementazione di riferimento 1D.

Architettura a tre livelli geometrici
(g0 = essenza invariante, gx = operativita' vincolata, gy = novita' controllata)
come sistema di reazione-diffusione stocastica.

    dF/dt = D_g * lap(F) + R_g(F, x, t)

    g0:  R = 0                                    (conservazione)
    gx:  R = alpha(x,t) * (F_input - F)            (adattamento)
    gy:  R = beta*F*(1 - F/K) + gamma*eta(x,t)     (creativita' vincolata)

Diagnostiche:
  - V(F) = ||F - F_g0||^2_L2          (funzione di Lyapunov)
  - M(t) = integrale di F             (misura totale, invariante n = v)
  - qualita' / continuita' / fedelta' (metriche sezione 9 del preprint)

Autore del modello: dr. Bulla Francesco
Dipendenze: numpy, matplotlib
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------
# Parametri
# ----------------------------------------------------------------------
@dataclass
class Params:
    # dominio spaziale (spazio astratto delle rappresentazioni cognitive)
    L: float = 1.0          # lunghezza del dominio Omega
    N: int = 256            # nodi della griglia
    periodic: bool = True   # Omega come toro (varieta' compatta senza bordo)

    # tempo
    T: float = 2.0
    dt: float | None = None  # None -> passo stabile calcolato automaticamente
    save_every: int = 20

    # coefficienti di diffusione per livello
    D_g0: float = 1.0e-3
    D_gx: float = 5.0e-4
    D_gy: float = 1.0e-4

    # accoppiamento con l'input (livello gx)
    alpha0: float = 1.5      # ampiezza del coefficiente di adattamento
    input_center: float = 0.65
    input_width: float = 0.05
    input_amp: float = 1.0
    input_freq: float = 2.0  # oscillazione temporale dell'input

    # novita' controllata (livello gy)
    beta: float = 1.2        # crescita creativa
    K: float = 1.0           # capacita' massima di novita'
    gamma: float = 0.02      # intensita' del rumore creativo
    noise_corr_len: float = 0.03  # lunghezza di correlazione (rumore colorato)

    # vincolo di compatibilita' gy -> g0/gx (proiezione sul cono ammissibile)
    novelty_budget: float = 0.25  # quota massima di deviazione da g0 tollerata

    seed: int = 0
    conserve_mass: bool = True    # riproiezione per imporre n = v

    def __post_init__(self):
        if self.dt is None:
            dx = self.L / self.N
            Dmax = max(self.D_g0, self.D_gx, self.D_gy)
            # CFL esplicito 1D: dt <= dx^2 / (2 D), con margine
            self.dt = 0.2 * dx * dx / max(Dmax, 1e-12)


# ----------------------------------------------------------------------
# Operatori sul dominio
# ----------------------------------------------------------------------
class Domain:
    """Griglia 1D su Omega, periodica (toro) o con bordi di Neumann."""

    def __init__(self, p: Params):
        self.p = p
        self.dx = p.L / p.N
        self.x = np.linspace(0.0, p.L, p.N, endpoint=p.periodic is False)
        if p.periodic:
            self.x = np.arange(p.N) * self.dx

    def laplacian(self, F: np.ndarray) -> np.ndarray:
        if self.p.periodic:
            return (np.roll(F, -1) - 2.0 * F + np.roll(F, 1)) / self.dx ** 2
        Fp = np.empty(F.size + 2)
        Fp[1:-1] = F
        Fp[0] = F[0]      # Neumann omogeneo: flusso nullo al bordo
        Fp[-1] = F[-1]
        return (Fp[2:] - 2.0 * Fp[1:-1] + Fp[:-2]) / self.dx ** 2

    def integral(self, F: np.ndarray) -> float:
        return float(np.sum(F) * self.dx)

    def l2(self, F: np.ndarray) -> float:
        return float(np.sqrt(np.sum(F ** 2) * self.dx))


# ----------------------------------------------------------------------
# Livelli geometrici
# ----------------------------------------------------------------------
class GeometricLevel:
    """Interfaccia comune: coefficiente di diffusione + termine di reazione."""

    name = "g"
    D = 0.0

    def reaction(self, F, t, dom, rng):
        return np.zeros_like(F)


class G0(GeometricLevel):
    """Essenza perfetta: nessuna reazione, dinamica puramente conservativa."""

    name = "g0"

    def __init__(self, p: Params, F0: np.ndarray, frozen: bool = True):
        self.D = 0.0 if frozen else p.D_g0
        self.frozen = frozen
        self.F_ref = F0.copy()   # F_g0(x): distribuzione ideale


class GX(GeometricLevel):
    """Operativita' vincolata: si adatta all'input tramite alpha(x,t)."""

    name = "gx"

    def __init__(self, p: Params):
        self.p = p
        self.D = p.D_gx

    def alpha(self, x, t):
        # finestra di attenzione mobile: l'adattamento non e' uniforme su Omega
        c = self.p.input_center + 0.15 * np.sin(2 * np.pi * self.p.input_freq * t)
        return self.p.alpha0 * np.exp(-((x - c) ** 2) / (2 * self.p.input_width ** 2))

    def F_input(self, x, t):
        c = self.p.input_center + 0.15 * np.sin(2 * np.pi * self.p.input_freq * t)
        return self.p.input_amp * np.exp(-((x - c) ** 2) / (2 * self.p.input_width ** 2))

    def reaction(self, F, t, dom, rng):
        x = dom.x
        return self.alpha(x, t) * (self.F_input(x, t) - F)


class GY(GeometricLevel):
    """Novita' controllata: logistica + rumore colorato, entro un budget."""

    name = "gy"

    def __init__(self, p: Params):
        self.p = p
        self.D = p.D_gy

    def colored_noise(self, dom, rng):
        w = rng.standard_normal(dom.p.N)
        # filtro gaussiano in Fourier -> rumore con lunghezza di correlazione
        k = np.fft.rfftfreq(dom.p.N, d=dom.dx) * 2 * np.pi
        filt = np.exp(-0.5 * (k * self.p.noise_corr_len) ** 2)
        w = np.fft.irfft(np.fft.rfft(w) * filt, n=dom.p.N)
        s = w.std()
        return w / s if s > 0 else w

    def reaction(self, F, t, dom, rng):
        logistic = self.p.beta * F * (1.0 - F / self.p.K)
        noise = self.p.gamma * self.colored_noise(dom, rng) / np.sqrt(dom.p.dt)
        return logistic + noise


# ----------------------------------------------------------------------
# Simulatore
# ----------------------------------------------------------------------
@dataclass
class History:
    t: list = field(default_factory=list)
    V: list = field(default_factory=list)      # Lyapunov
    mass: list = field(default_factory=list)   # misura totale
    quality: list = field(default_factory=list)
    continuity: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)


class FrammentoDelVeloce:
    def __init__(self, p: Params, evolve_g0: bool = False):
        self.p = p
        self.dom = Domain(p)
        self.rng = np.random.default_rng(p.seed)

        self.F = self.essence(self.dom.x)
        self.g0 = G0(p, self.F, frozen=not evolve_g0)
        self.gx = GX(p)
        self.gy = GY(p)
        self.M0 = self.dom.integral(self.F)
        self.hist = History()

    # -- condizione iniziale: l'essenza g0 ------------------------------
    def essence(self, x):
        """Traccia mnestica di riferimento: due modi sovrapposti."""
        f = (np.exp(-((x - 0.30) ** 2) / (2 * 0.04 ** 2))
             + 0.6 * np.exp(-((x - 0.55) ** 2) / (2 * 0.07 ** 2)))
        return f

    # -- diagnostiche ----------------------------------------------------
    def lyapunov(self) -> float:
        d = self.F - self.g0.F_ref
        return float(np.sum(d ** 2) * self.dom.dx)

    def quality(self) -> float:
        """Correlazione normalizzata tra stato corrente ed essenza g0."""
        a = self.F - self.F.mean()
        b = self.g0.F_ref - self.g0.F_ref.mean()
        den = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / den) if den > 0 else 0.0

    def continuity(self, F_prev: np.ndarray) -> float:
        """Coerenza temporale: 1 - variazione relativa per unita' di tempo."""
        num = self.dom.l2(self.F - F_prev)
        den = self.dom.l2(F_prev) + 1e-12
        return float(np.exp(-num / den))

    # -- vincolo strutturale gy ------------------------------------------
    def project_novelty(self, F):
        """gy puo' innovare solo entro un budget di deviazione da g0."""
        d = F - self.g0.F_ref
        norm_d = self.dom.l2(d)
        budget = self.p.novelty_budget * self.dom.l2(self.g0.F_ref)
        if norm_d > budget > 0:
            d *= budget / norm_d
        return self.g0.F_ref + d

    def project_mass(self, F):
        """Imposizione dell'invariante n = v: integrale di F costante."""
        M = self.dom.integral(F)
        if abs(M) < 1e-12:
            return F
        return F * (self.M0 / M)

    # -- integrazione temporale ------------------------------------------
    def step(self, t):
        p, dom = self.p, self.dom
        F = self.F

        # diffusione: contributo pesato dei tre livelli
        D_eff = self.g0.D + self.gx.D + self.gy.D
        lap = dom.laplacian(F)

        R = (self.gx.reaction(F, t, dom, self.rng)
             + self.gy.reaction(F, t, dom, self.rng))

        F_new = F + p.dt * (D_eff * lap + R)
        F_new = np.clip(F_new, 0.0, None)           # densita' non negativa
        F_new = self.project_novelty(F_new)         # vincolo gy -> g0
        if p.conserve_mass:
            F_new = self.project_mass(F_new)        # invariante n = v

        self.F = F_new

    def run(self, verbose: bool = True):
        p = self.p
        n_steps = int(p.T / p.dt)
        F_prev = self.F.copy()

        for i in range(n_steps):
            t = i * p.dt
            if i % p.save_every == 0:
                self.hist.t.append(t)
                self.hist.V.append(self.lyapunov())
                self.hist.mass.append(self.dom.integral(self.F))
                self.hist.quality.append(self.quality())
                self.hist.continuity.append(self.continuity(F_prev))
                self.hist.snapshots.append(self.F.copy())
                F_prev = self.F.copy()
            self.step(t)

        if verbose:
            self.report(n_steps)
        return self.hist

    # -- fedelta': resistenza a una perturbazione di riconsolidamento -----
    def fidelity_test(self, amplitude: float = 0.5, relax_T: float = 0.5) -> float:
        """Perturba il ricordo, lascia rilassare, misura il recupero."""
        F_before = self.F.copy()
        pert = amplitude * np.exp(-((self.dom.x - 0.45) ** 2) / (2 * 0.03 ** 2))
        self.F = np.clip(self.F + pert, 0.0, None)
        for i in range(int(relax_T / self.p.dt)):
            self.step(self.p.T + i * self.p.dt)
        err = self.dom.l2(self.F - F_before) / (self.dom.l2(F_before) + 1e-12)
        return float(np.exp(-err))

    def report(self, n_steps):
        h = self.hist
        drift = abs(h.mass[-1] - self.M0) / max(abs(self.M0), 1e-12)
        dV = np.diff(h.V)
        monotone = float(np.mean(dV <= 1e-12)) if dV.size else 1.0
        print(f"passi                : {n_steps}  (dt = {self.p.dt:.3e})")
        print(f"massa iniziale M0    : {self.M0:.6f}")
        print(f"deriva di massa      : {drift:.3e}   (invariante n = v)")
        print(f"V finale             : {h.V[-1]:.6e}")
        print(f"frazione dV/dt <= 0  : {monotone:.3f}")
        print(f"qualita' finale      : {h.quality[-1]:.4f}")
        print(f"continuita' media    : {np.mean(h.continuity):.4f}")
        print(f"fedelta'             : {self.fidelity_test():.4f}")


# ----------------------------------------------------------------------
# Grafici
# ----------------------------------------------------------------------
def plot(sim: FrammentoDelVeloce, path: str = "frammento_del_veloce.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, x = sim.hist, sim.dom.x
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    ax[0, 0].plot(x, sim.g0.F_ref, "k--", lw=2, label="g0 (essenza)")
    idx = np.linspace(0, len(h.snapshots) - 1, 6).astype(int)
    for j in idx:
        ax[0, 0].plot(x, h.snapshots[j], lw=1, alpha=0.8, label=f"t={h.t[j]:.2f}")
    ax[0, 0].set_title("Diffusione del Frammento")
    ax[0, 0].set_xlabel("x")
    ax[0, 0].legend(fontsize=7)

    ax[0, 1].plot(h.t, h.V)
    ax[0, 1].set_title(r"Funzione di Lyapunov $V=\|F-F_{g0}\|^2$")
    ax[0, 1].set_xlabel("t")

    ax[1, 0].plot(h.t, np.array(h.mass) / sim.M0)
    ax[1, 0].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[1, 0].set_title("Misura totale normalizzata (n = v)")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].set_ylim(0.9, 1.1)

    ax[1, 1].plot(h.t, h.quality, label="qualita'")
    ax[1, 1].plot(h.t, h.continuity, label="continuita'")
    ax[1, 1].set_title("Metriche")
    ax[1, 1].set_xlabel("t")
    ax[1, 1].set_ylim(0, 1.05)
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"grafico salvato      : {path}")
    return np.array(h.snapshots)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Frammento del veloce - simulatore 1D")
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.02, help="rumore creativo")
    ap.add_argument("--beta", type=float, default=1.2, help="crescita creativa")
    ap.add_argument("--budget", type=float, default=0.25, help="budget di novita' gy")
    ap.add_argument("--no-mass", action="store_true", help="disattiva n = v")
    ap.add_argument("--evolve-g0", action="store_true",
                    help="lascia diffondere g0 (regime D_g0 >> D_gx del preprint)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", type=str, default="")
    a = ap.parse_args()

    p = Params(T=a.T, N=a.N, gamma=a.gamma, beta=a.beta,
               novelty_budget=a.budget, conserve_mass=not a.no_mass, seed=a.seed)
    sim = FrammentoDelVeloce(p, evolve_g0=a.evolve_g0)
    sim.run()
    if a.plot:
        plot(sim, a.plot)


if __name__ == "__main__":
    main()
