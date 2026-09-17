"""Studi di robustezza: sweep dei parametri e ablazione dei livelli.

- sweep: griglia di configurazioni (diffusioni, alpha, beta, gamma, K,
  rumore bianco/OU, protocolli stimolo/rilassamento). Per ogni config salva:
  massa g0 finale, qualita', continuita', fedelta', max(dV/dt),
  frazione dV/dt<=0, novita' finale, tempo di recupero dopo perturbazione.
- ablazione: confronto con baseline (solo diffusione, g0+gx, completo,
  completo senza vincolo, senza rumore, rumore eccessivo).

Uso:
    python -m src sweep --N 48 --T 0.15
    python -m src ablazione --N 48 --T 0.15
"""

from __future__ import annotations

import csv
import os
from dataclasses import replace

import numpy as np

try:
    from src.frammento_2d import (
        Param, derivata_numerica, griglia, input_field, metriche, simula,
    )
except ImportError:  # lancio diretto come script dentro src/
    from frammento_2d import (
        Param, derivata_numerica, griglia, input_field, metriche, simula,
    )

EPS = 1e-12


# ---------------------------------------------------------------------------
# Utilita'
# ---------------------------------------------------------------------------
def safe_corr(A: np.ndarray, B: np.ndarray) -> float:
    """Correlazione di Pearson robusta (1.0 se i campi coincidono)."""
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    if np.allclose(A, B):
        return 1.0
    sa, sb = A.std(), B.std()
    if sa < EPS or sb < EPS:
        return 0.0
    c = float(np.corrcoef(A, B)[0, 1])
    return c if np.isfinite(c) else 0.0


def tempo_recupero(p: Param, base: dict, ampiezza: float = 0.5,
                   T_rec: float = 0.08, salva_ogni: int = 5,
                   frazione: float = 0.05) -> tuple[float, bool]:
    """Twin experiment: gemello imperturbato vs gemello perturbato (bump).

    Entrambi ripartono dallo stato finale di `base` in protocollo
    'rilassamento' deterministico; si misura il primo istante in cui
    l'energia della perturbazione ||Fp - Fu|| scende sotto `frazione`
    del suo valore iniziale. Ritorna (t_rec, recuperato).
    """
    X, Y = griglia(p)
    bump = ampiezza * np.exp(-((X - 0.5 * p.L) ** 2 + (Y - 0.5 * p.L) ** 2)
                             / (2 * (0.03 * p.L) ** 2))
    comuni = dict(T=T_rec, protocollo="rilassamento",
                  stocastico=False, salva_ogni=salva_ogni)
    init_u = {"F0": base["F0"][-1], "Fx": base["Fx"][-1], "Fy": base["Fy"][-1]}
    init_p = {"F0": base["F0"][-1], "Fx": base["Fx"][-1] + bump,
              "Fy": base["Fy"][-1]}
    su = simula(p, stato_iniziale=init_u, **comuni)
    sp = simula(p, stato_iniziale=init_p, **comuni)
    e0 = float(np.sqrt(np.sum(bump ** 2)))
    if e0 < EPS:
        return 0.0, True
    for t, Au, Ap in zip(sp["t"], su["Fx"], sp["Fx"]):
        r = float(np.sqrt(np.sum((np.asarray(Ap) - np.asarray(Au)) ** 2))) / e0
        if r < frazione:
            return float(t), True
    return float(T_rec), False


# ---------------------------------------------------------------------------
# Valutazione di una configurazione
# ---------------------------------------------------------------------------
def valuta(p: Param, T: float = 0.15, salva_ogni: int = 20, seed: int = 7,
           rumore_bianco: bool = True, protocollo: str = "stimolo",
           T_rec: float = 0.30) -> dict:
    """Esegue coppia deterministica/stocastica + twin di recupero.

    Ritorna metriche scalari: massa g0 finale, qualita' finale, continuita'
    media, fedelta' della novita' gy (det vs stoc: il rumore entra solo in
    gy, quindi il confronto su gx sarebbe identicamente 1), max(dV/dt),
    frazione dV/dt<=0, novita' finale, gate medio finale, tempo di recupero.
    """
    p = replace(p, seed=seed)
    kw = dict(T=T, salva_ogni=salva_ogni, rumore_bianco=rumore_bianco,
              protocollo=protocollo)
    snap_s = simula(p, stocastico=True, **kw)
    snap_d = simula(p, stocastico=False, **kw)
    m = metriche(snap_s)
    d = snap_s["diag"]
    dV = derivata_numerica(d["t"], d["V"])
    t_rec, recuperato = tempo_recupero(p, snap_d, T_rec=T_rec)
    return {
        "massa_g0_fin": float(d["massa0"][-1]),
        "qualita_fin": float(m["qualita"][-1]),
        "continuita_media": float(m["continuita"].mean()),
        "fedelta": safe_corr(snap_d["Fy"][-1], snap_s["Fy"][-1]),
        "max_dVdt": float(np.max(dV)) if dV.size else 0.0,
        "fraz_dVdt_nonpos": float(np.mean(dV <= 1e-9)) if dV.size else 1.0,
        "novita_fin": float(d["novita"][-1]),
        "gate_fin": float(d["gate_medio"][-1]),
        "t_rec": t_rec,
        "recuperato": recuperato,
    }


# ---------------------------------------------------------------------------
# Configurazioni
# ---------------------------------------------------------------------------
def config_sweep() -> list[dict]:
    """Griglia di robustezza: una variazione rilevante per riga."""
    base = dict(N=48, D0=0.05, Dx=0.01, Dy=0.002, alpha=3.0, kappa_x=0.6,
                beta=0.9, K=1.2, gamma=0.02, kappa_y=0.25, seed=7)
    return [
        {"nome": "A base (stimolo, bianco)",
         "param": dict(base), "sim": {}},
        {"nome": "B rumore nullo (gamma=0)",
         "param": {**base, "gamma": 0.0}, "sim": {}},
        {"nome": "C rumore alto (gamma=0.15)",
         "param": {**base, "gamma": 0.15}, "sim": {}},
        {"nome": "D rumore OU colorato",
         "param": dict(base), "sim": {"rumore_bianco": False}},
        {"nome": "E creativita' alta (beta=2.0)",
         "param": {**base, "beta": 2.0}, "sim": {}},
        {"nome": "F protocollo rilassamento",
         "param": dict(base), "sim": {"protocollo": "rilassamento"}},
        {"nome": "G adattamento forte (alpha=8.0)",
         "param": {**base, "alpha": 8.0}, "sim": {}},
    ]


def config_ablazione() -> list[dict]:
    """Baseline di controllo: cosa aggiunge ogni livello/vincolo."""
    base = dict(N=48, D0=0.05, Dx=0.01, Dy=0.002, alpha=3.0, kappa_x=0.6,
                beta=0.9, K=1.2, gamma=0.02, kappa_y=0.25, seed=7)
    return [
        {"nome": "1 solo diffusione",
         "param": {**base, "alpha": 0.0, "kappa_x": 0.0,
                   "beta": 0.0, "gamma": 0.0},
         "sim": {"stocastico": False}},
        {"nome": "2 g0+gx (senza gy)",
         "param": {**base, "beta": 0.0, "gamma": 0.0},
         "sim": {"stocastico": False}},
        {"nome": "3 completo",
         "param": dict(base), "sim": {}},
        {"nome": "4 completo senza vincolo (ky=0, K=5)",
         "param": {**base, "kappa_y": 0.0, "K": 5.0}, "sim": {}},
        {"nome": "5 completo senza rumore",
         "param": dict(base), "sim": {"stocastico": False}},
        {"nome": "6 rumore eccessivo (gamma=0.25)",
         "param": {**base, "gamma": 0.25}, "sim": {}},
    ]


# ---------------------------------------------------------------------------
# Esecuzione e report
# ---------------------------------------------------------------------------
COLONNE_SWEEP = ["config", "massa_g0_fin", "qualita_fin", "continuita_media",
                 "fedelta", "max_dVdt", "fraz_dVdt_nonpos", "novita_fin",
                 "t_rec", "recuperato"]


def _adattamento(p: Param, snap: dict) -> float:
    Fin = input_field(p, float(snap["t"][-1]))
    return safe_corr(snap["Fx"][-1], Fin)


def esegui(configs: list[dict], T: float, salva_ogni: int, seed: int,
           con_adattamento: bool = False) -> list[dict]:
    righe = []
    for cfg in configs:
        p = Param(**cfg["param"])
        r = valuta(p, T=T, salva_ogni=salva_ogni, seed=seed,
                   rumore_bianco=cfg["sim"].get("rumore_bianco", True),
                   protocollo=cfg["sim"].get("protocollo", "stimolo"))
        r["config"] = cfg["nome"]
        if con_adattamento:
            snap = simula(p, T=T, salva_ogni=salva_ogni,
                          stocastico=cfg["sim"].get("stocastico", True),
                          rumore_bianco=cfg["sim"].get("rumore_bianco", True),
                          protocollo=cfg["sim"].get("protocollo", "stimolo"))
            r["adattamento"] = _adattamento(p, snap)
        print(f"[{r['config']}] Q={r['qualita_fin']:.3f} "
              f"fed={r['fedelta']:.4f} nov={r['novita_fin']:.4f} "
              f"dV<=0: {r['fraz_dVdt_nonpos'] * 100:.0f}% "
              f"t_rec={r['t_rec']:.3f}{'' if r['recuperato'] else '+'}")
        righe.append(r)
    return righe


def salva_csv(righe: list[dict], colonne: list[str], path: str):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=colonne, extrasaction="ignore")
        w.writeheader()
        w.writerows(righe)
    print("salvato:", path)


def tabella_md(righe: list[dict], colonne: list[str]) -> str:
    def fmt(v):
        if isinstance(v, bool):
            return "si" if v else "no"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    head = "| " + " | ".join(colonne) + " |"
    sep = "|" + "|".join(["---"] * len(colonne)) + "|"
    body = ["| " + " | ".join(fmt(r.get(c, "")) for c in colonne) + " |"
            for r in righe]
    return "\n".join([head, sep] + body)


def fig_ablazione(righe: list[dict], path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nomi = [r["config"] for r in righe]
    coppie = [("qualita_fin", "qualita'"), ("continuita_media", "continuita'"),
              ("fedelta", "fedelta'"), ("adattamento", "adattamento")]
    x = np.arange(len(nomi))
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    for j, (chiave, label) in enumerate(coppie):
        vals = [max(0.0, min(1.0, r[chiave])) for r in righe]
        ax.bar(x + (j - 1.5) * 0.18, vals, 0.17, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(nomi, rotation=14, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablazione: contributo di livelli, vincoli e rumore")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("salvato:", path)


def main_sweep(N: int = 48, T: float = 0.15, out_dir: str = "output",
               seed: int = 7):
    os.makedirs(out_dir, exist_ok=True)
    configs = config_sweep()
    for c in configs:
        c["param"]["N"] = N
    righe = esegui(configs, T=T, salva_ogni=20, seed=seed)
    salva_csv(righe, COLONNE_SWEEP, os.path.join(out_dir, "studio_parametri.csv"))
    md = "# Studio dei parametri\n\n" + tabella_md(righe, COLONNE_SWEEP) + "\n"
    with open(os.path.join(out_dir, "studio_parametri.md"), "w") as f:
        f.write(md)
    print("salvato:", os.path.join(out_dir, "studio_parametri.md"))
    return righe


def main_ablazione(N: int = 48, T: float = 0.15, out_dir: str = "output",
                   seed: int = 7):
    os.makedirs(out_dir, exist_ok=True)
    configs = config_ablazione()
    for c in configs:
        c["param"]["N"] = N
    righe = esegui(configs, T=T, salva_ogni=20, seed=seed,
                   con_adattamento=True)
    colonne = COLONNE_SWEEP + ["adattamento"]
    salva_csv(righe, colonne, os.path.join(out_dir, "ablazione.csv"))
    md = "# Ablazione dei livelli\n\n" + tabella_md(righe, colonne) + "\n"
    with open(os.path.join(out_dir, "ablazione.md"), "w") as f:
        f.write(md)
    print("salvato:", os.path.join(out_dir, "ablazione.md"))
    fig_ablazione(righe, os.path.join(out_dir, "fig08_ablazione.png"))
    return righe
