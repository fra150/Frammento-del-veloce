"""Demo e figure per il modello 'Frammento del veloce'.

Uso:
    python -m src.demo_figure
    python demo_figure.py

Produce in output/ (o cartella indicata):
    fig01_tre_livelli.png      campi g0, gx, gy a t finale
    fig02_evoluzione.png       snapshot temporali di gx e gy
    fig03_lyapunov_massa.png   V(t), masse, novita'
    fig04_metriche.png         qualita' / continuita'
    fig05_turing_gy.png        novita' controllata (pattern)
    fig06_invariante_nv.png    verifica n = v (scala g0)
    fig07_lfp.png              LFP sintetico theta-gamma

Dipendenze: numpy, matplotlib.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from src.frammento_2d import (
        Param, simula, metriche, turing_gy, invariante_nv,
        esperimento_diffusione, lfp_sintetico, derivata_numerica,
        input_field,
    )
except ImportError:  # lancio diretto come script dentro src/
    from frammento_2d import (
        Param, simula, metriche, turing_gy, invariante_nv,
        esperimento_diffusione, lfp_sintetico, derivata_numerica,
        input_field,
    )

OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)


def _salva(fig, nome):
    path = os.path.join(OUT, nome)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("salvato:", path)
    return path


def _cmap_campo():
    return "magma"


# ---------------------------------------------------------------------------
# 1. Tre livelli a t finale
# ---------------------------------------------------------------------------
def fig_tre_livelli(snap):
    F0 = snap["F0"][-1]
    Fx = snap["Fx"][-1]
    Fy = snap["Fy"][-1]
    Fin = input_field(snap["param"], float(snap["t"][-1]))
    fig, ax = plt.subplots(1, 4, figsize=(14.2, 3.4))
    campi = [
        (F0, r"$g_0$  essenza (invariante)"),
        (Fin, r"input $x(t)$"),
        (Fx, r"$g_x$  operativita' vincolata"),
        (Fy, r"$g_y$  novita' controllata"),
    ]
    for a, (Z, titolo) in zip(ax, campi):
        im = a.imshow(Z.T, origin="lower", cmap=_cmap_campo(),
                      extent=[0, 1, 0, 1])
        a.set_title(titolo, fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle("Frammento del veloce  -  architettura a tre livelli",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    return _salva(fig, "fig01_tre_livelli.png")


# ---------------------------------------------------------------------------
# 2. Evoluzione temporale
# ---------------------------------------------------------------------------
def fig_evoluzione(snap):
    idx = np.linspace(0, len(snap["t"]) - 1, 5, dtype=int)
    fig, ax = plt.subplots(2, 5, figsize=(14.5, 5.6))
    for j, i in enumerate(idx):
        ax[0, j].imshow(snap["Fx"][i].T, origin="lower", cmap=_cmap_campo(),
                        extent=[0, 1, 0, 1])
        ax[0, j].set_title(rf"$g_x$  t = {snap['t'][i]:.3f}", fontsize=9)
        ax[0, j].set_xticks([])
        ax[0, j].set_yticks([])
        ax[1, j].imshow(snap["Fy"][i].T, origin="lower", cmap="inferno",
                        extent=[0, 1, 0, 1])
        ax[1, j].set_title(rf"$g_y$  t = {snap['t'][i]:.3f}", fontsize=9)
        ax[1, j].set_xticks([])
        ax[1, j].set_yticks([])
    ax[0, 0].set_ylabel(r"$g_x$ adattamento", fontsize=10)
    ax[1, 0].set_ylabel(r"$g_y$ creativita'", fontsize=10)
    fig.suptitle("Diffusione del ricordo: adattamento (gx) e novita' (gy)",
                 fontsize=13)
    fig.tight_layout()
    return _salva(fig, "fig02_evoluzione.png")


# ---------------------------------------------------------------------------
# 3. Lyapunov, masse, novita'
# ---------------------------------------------------------------------------
def fig_diagnostica(snap):
    d = snap["diag"]
    dV = derivata_numerica(d["t"], d["V"])
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.6))

    ax[0].plot(d["t"], d["V"], color="#1f4e79", lw=2)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel(r"$V(\mathcal{F})$")
    ax[0].set_title("funzione di Lyapunov")
    ax[0].grid(alpha=0.3)

    ax[1].plot(d["t"], d["massa0"], label=r"massa $g_0$", lw=2)
    ax[1].plot(d["t"], d["massaX"], label=r"massa $g_x$", lw=2)
    ax[1].plot(d["t"], d["novita"], label=r"novita' $g_y$", lw=2)
    ax[1].set_xlabel("t")
    ax[1].set_title("misure (invariante n)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].plot(d["t"], dV, color="#8b1e3f", lw=1.5)
    ax[2].axhline(0, color="k", ls="--", lw=0.8)
    ax[2].set_xlabel("t")
    ax[2].set_ylabel(r"$dV/dt$")
    ax[2].set_title(r"stabilita': $dV/dt \leq 0$")
    ax[2].grid(alpha=0.3)

    fig.suptitle("Stabilita' e conservazione del Frammento", fontsize=13)
    fig.tight_layout()
    return _salva(fig, "fig03_lyapunov_massa.png")


# ---------------------------------------------------------------------------
# 4. Metriche di qualita' e continuita'
# ---------------------------------------------------------------------------
def fig_metriche(snap):
    m = metriche(snap)
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.5))
    ax[0].plot(m["t"], m["qualita"], color="#2a6f4e", lw=2)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("qualita'")
    ax[0].set_title(r"somiglianza di $g_x$ all'essenza $g_0$")
    ax[0].set_ylim(-0.2, 1.05)
    ax[0].grid(alpha=0.3)

    ax[1].plot(m["t"], m["continuita"], color="#6b3fa0", lw=2)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("continuita'")
    ax[1].set_title("coerenza temporale (coseno tra frame)")
    ax[1].set_ylim(0.0, 1.05)
    ax[1].grid(alpha=0.3)

    fig.suptitle("Metriche di qualita' e continuita' del ricordo", fontsize=13)
    fig.tight_layout()
    return _salva(fig, "fig04_metriche.png")


# ---------------------------------------------------------------------------
# 5. Pattern di Turing in gy (novita' controllata)
# ---------------------------------------------------------------------------
def fig_turing(p):
    out = turing_gy(p, passi=9000, dt=2.5e-4)
    fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.5))
    ax[0].imshow(out["mask"].T, origin="lower", cmap="gray", extent=[0, 1, 0, 1])
    ax[0].set_title(r"supporto di $g_0$ (vincolo strutturale)")
    ax[1].imshow(out["a"].T, origin="lower", cmap="inferno", extent=[0, 1, 0, 1])
    ax[1].set_title(r"attivatore $g_y$ (novita')")
    ax[2].imshow(out["h"].T, origin="lower", cmap="viridis", extent=[0, 1, 0, 1])
    ax[2].set_title("inibitore (stabilizza i pattern)")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Novita' controllata: pattern emergenti vincolati da g0",
                 fontsize=13)
    fig.tight_layout()
    return _salva(fig, "fig05_turing_gy.png")


# ---------------------------------------------------------------------------
# 6. Invariante n = v
# ---------------------------------------------------------------------------
def fig_invariante(p):
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.7))
    colori = {"g0": "#1f4e79", "gx": "#2a6f4e", "gy": "#8b1e3f"}
    Ds = {"g0": p.D0, "gx": p.Dx, "gy": p.Dy}

    for nome, D in Ds.items():
        e = esperimento_diffusione(p, D, T=0.05)
        ax[0].plot(e["t"], e["r2"], "o-", ms=3, color=colori[nome],
                   label=fr"{nome}: D = {D:.3f}")
        ax[0].plot(e["t"], 4 * D * e["t"] + e["r2"][0], "--",
                   color=colori[nome], alpha=0.55, lw=1)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel(r"$\langle r^2\rangle$")
    ax[0].set_title(r"diffusione: $\langle r^2\rangle = 4Dt$")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    nomi, vtilde, Dstim = [], [], []
    for nome, D in Ds.items():
        inv = invariante_nv(p, D, T=0.05)
        nomi.append(nome)
        vtilde.append(inv["v_tilde"])
        Dstim.append(inv["D_stimato"] / D)

    x = np.arange(len(nomi))
    ax[1].bar(x - 0.18, vtilde, 0.36, label=r"$v / D_{g0}$  (scala n = v)",
              color="#1f4e79")
    ax[1].bar(x + 0.18, Dstim, 0.36, label=r"$D_{\mathrm{stim}} / D$",
              color="#c47b17")
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(nomi)
    ax[1].set_title(r"invarianza $n=v$ (metrica di $g_0$)")
    ax[1].legend(fontsize=8)
    ax[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Principio n = v, v = n  (scala fissata da g0)", fontsize=13)
    fig.tight_layout()
    return _salva(fig, "fig06_invariante_nv.png")


# ---------------------------------------------------------------------------
# 7. LFP sintetico
# ---------------------------------------------------------------------------
def fig_lfp(snap):
    sig = lfp_sintetico(snap["param"], snap["Fy"][-1], durata=1.5)
    t, y, fs = sig["t"], sig["lfp"], sig["fs"]
    Y = np.fft.rfft(y * np.hanning(y.size))
    f = np.fft.rfftfreq(y.size, 1.0 / fs)

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.5))
    ax[0].plot(t, y, color="#333", lw=0.7)
    ax[0].set_xlim(0, 0.6)
    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel("LFP")
    ax[0].set_title("traccia sintetica (theta + gamma gated)")
    ax[0].grid(alpha=0.3)

    ax[1].plot(f, np.abs(Y), color="#8b1e3f", lw=1.4)
    ax[1].axvline(6, color="#1f4e79", ls="--", lw=1, label="theta 6 Hz")
    ax[1].axvline(45, color="#c47b17", ls="--", lw=1, label="gamma 45 Hz")
    ax[1].set_xlim(0, 80)
    ax[1].set_xlabel("f [Hz]")
    ax[1].set_ylabel("|Y|")
    ax[1].set_title("spettro (correlato neurofisiologico)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.suptitle("Protocollo neurofisiologico simulato (EEG/LFP)", fontsize=13)
    fig.tight_layout()
    return _salva(fig, "fig07_lfp.png")


def main():
    p = Param(N=80, D0=0.008, Dx=0.012, Dy=0.003,
              alpha=4.0, kappa_x=0.8, beta=1.4, K=1.15,
              gamma=0.015, kappa_y=0.18, seed=7)
    print("dt stabile =", p.dt_stabile())
    print("integrazione in corso...")
    snap = simula(p, T=0.22, protocollo="stimolo", salva_ogni=40, stocastico=True)
    d = snap["diag"]
    print(f"massa g0 : {d['massa0'][0]:.4f} -> {d['massa0'][-1]:.4f}")
    print(f"novita'  : {d['novita'][-1]:.4f}")
    print(f"V        : {d['V'][0]:.4f} -> {d['V'][-1]:.4f}")

    fig_tre_livelli(snap)
    fig_evoluzione(snap)
    fig_diagnostica(snap)
    fig_metriche(snap)
    fig_turing(p)
    fig_invariante(p)
    fig_lfp(snap)
    print("fine.")


if __name__ == "__main__":
    main()
