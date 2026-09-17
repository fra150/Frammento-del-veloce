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
                   T_rec: float = 1.00, salva_ogni: int = 2,
                   frazione: float = 0.05) -> tuple[float, bool]:
    """Twin experiment: gemello imperturbato vs gemello perturbato (bump).

    Entrambi ripartono dallo stato finale di `base` in protocollo
    'rilassamento' deterministico; si misura il primo istante in cui
    l'energia della perturbazione ||Fp - Fu|| scende sotto `frazione`
    del suo valore iniziale. Ritorna (t_rec, recuperato).

    T_rec=1.0 (scala ~3/(alpha+kappa_x)): con T_rec=0.30 quasi nessuna
    config recuperava (tasso ~3.6 -> tau~0.28, servono ~0.8 per il 5%).
    L'attraversamento della soglia e' interpolato linearmente tra i due
    snapshot a cavallo (non il primo punto di griglia), con snapshot fini
    (salva_ogni=2). Resta identico tra config che condividono lo stesso
    sottosistema Fx deterministico (atteso per disegno, non un artefatto).
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
    t_prev: float | None = None
    r_prev: float | None = None
    for t, Au, Ap in zip(sp["t"], su["Fx"], sp["Fx"]):
        r = float(np.sqrt(np.sum((np.asarray(Ap) - np.asarray(Au)) ** 2))) / e0
        if r < frazione:
            t = float(t)
            if t_prev is not None and r_prev is not None and r != r_prev:
                # interpolazione lineare dell'attraversamento di soglia
                f = (frazione - r_prev) / (r - r_prev)
                f = min(1.0, max(0.0, f))
                return t_prev + f * (t - t_prev), True
            return t, True
        t_prev, r_prev = float(t), r
    return float(T_rec), False


# ---------------------------------------------------------------------------
# Valutazione di una configurazione
# ---------------------------------------------------------------------------
def valuta(p: Param, T: float = 0.15, salva_ogni: int = 20, seed: int = 7,
           rumore_bianco: bool = True, protocollo: str = "stimolo",
           T_rec: float = 1.00, stocastico: bool = True) -> dict:
    """Esegue coppia deterministica/stocastica + twin di recupero.

    Ritorna metriche scalari: massa g0 finale, qualita' finale, continuita'
    media, fedelta' della novita' gy (det vs stoc: il rumore entra solo in
    gy, quindi il confronto su gx sarebbe identicamente 1), max(dV/dt),
    frazione dV/dt<=0, novita' finale, gate medio finale, tempo di recupero.
    Se stocastico=False, entrambe le repliche sono deterministiche
    (fedelta'=1 per costruzione: baseline senza rumore).
    """
    p = replace(p, seed=seed)
    kw = dict(T=T, salva_ogni=salva_ogni, rumore_bianco=rumore_bianco,
              protocollo=protocollo)
    snap_d = simula(p, stocastico=False, **kw)
    snap_s = simula(p, stocastico=True, **kw) if stocastico else snap_d
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


def _adattamento(p: Param, snap: dict, protocollo: str = "stimolo") -> float:
    """Avvicinamento di gx all'input: 1 - ||Fx-Fin|| / ||essenza-Fin||.

    0 = resta sull'essenza (nessun adattamento), 1 = replica perfetta
    dell'input. La vecchia versione (corr(Fx,Fin)) era degenere (~-0.04
    ovunque) perche' Fx e' dominata dall'essenza a due picchi mentre Fin
    e' un bump di Lissajous altrove: la correlazione resta ~0 per costruzione.
    La nuova misura la distanza relativa e cresce monotonicamente con alpha.
    Per protocollo 'rilassamento' l'input e' nullo (Fin=0).
    """
    import numpy as np_mod
    try:
        from src.frammento_2d import EPS as _EPS
    except ImportError:
        from frammento_2d import EPS as _EPS
    if protocollo == "rilassamento":
        Fin = np_mod.zeros_like(snap["Fx"][-1])
    else:
        Fin = input_field(p, float(snap["t"][-1]))
    Fx = np_mod.asarray(snap["Fx"][-1], dtype=float)
    ess = np_mod.asarray(snap["essenza"], dtype=float)
    num = float(np_mod.linalg.norm(Fx - Fin))
    den = float(np_mod.linalg.norm(ess - Fin)) + _EPS
    return float(1.0 - num / den)


def esegui(configs: list[dict], T: float, salva_ogni: int, seed: int,
           con_adattamento: bool = False) -> list[dict]:
    righe = []
    for cfg in configs:
        p = Param(**cfg["param"])
        r = valuta(p, T=T, salva_ogni=salva_ogni, seed=seed,
                   rumore_bianco=cfg["sim"].get("rumore_bianco", True),
                   protocollo=cfg["sim"].get("protocollo", "stimolo"),
                   stocastico=cfg["sim"].get("stocastico", True))
        r["config"] = cfg["nome"]
        if con_adattamento:
            proto = cfg["sim"].get("protocollo", "stimolo")
            snap = simula(p, T=T, salva_ogni=salva_ogni,
                          stocastico=cfg["sim"].get("stocastico", True),
                          rumore_bianco=cfg["sim"].get("rumore_bianco", True),
                          protocollo=proto)
            r["adattamento"] = _adattamento(p, snap, protocollo=proto)
        print(f"[{r['config']}] Q={r['qualita_fin']:.3f} "
              f"fed={r['fedelta']:.4f} nov={r['novita_fin']:.4f} "
              f"dV<=0: {r['fraz_dVdt_nonpos'] * 100:.0f}% "
              f"t_rec={r['t_rec']:.3f}{'' if r['recuperato'] else '+'}")
        righe.append(r)
    return righe


# ---------------------------------------------------------------------------
# Multi-seed: media ± std per distinguere effetto reale dal rumore
# ---------------------------------------------------------------------------
SEEDS_DEFAULT = (7, 11, 13, 21, 33)

METRICHE_NUMERICHE = ["massa_g0_fin", "qualita_fin", "continuita_media",
                      "fedelta", "max_dVdt", "fraz_dVdt_nonpos",
                      "novita_fin", "gate_fin", "t_rec"]


def valuta_multiseed(p: Param, seeds: tuple[int, ...] = SEEDS_DEFAULT,
                     T: float = 0.15, salva_ogni: int = 20,
                     rumore_bianco: bool = True, protocollo: str = "stimolo",
                     T_rec: float = 1.00,
                     con_adattamento: bool = False,
                     stocastico_adatt: bool = True,
                     stocastico: bool = True) -> dict:
    """Ripete `valuta` su piu' seed: media ± std + frazione recuperati.

    Necessario perche' con un solo seed una differenza 1.0000 vs 0.9994
    non e' distinguibile dal rumore. Ritorna chiavi *_mean / *_std,
    'recuperato_frac' e 'n_seed'.
    """
    campioni: dict[str, list[float]] = {k: [] for k in METRICHE_NUMERICHE}
    rec = 0
    adatt: list[float] = []
    for s in seeds:
        r = valuta(p, T=T, salva_ogni=salva_ogni, seed=int(s),
                   rumore_bianco=rumore_bianco, protocollo=protocollo,
                   T_rec=T_rec, stocastico=stocastico)
        for k in METRICHE_NUMERICHE:
            campioni[k].append(float(r[k]))
        rec += 1 if r["recuperato"] else 0
        if con_adattamento:
            ps = replace(p, seed=int(s))
            snap = simula(ps, T=T, salva_ogni=salva_ogni,
                          stocastico=stocastico_adatt,
                          rumore_bianco=rumore_bianco,
                          protocollo=protocollo)
            adatt.append(_adattamento(ps, snap, protocollo=protocollo))
    out: dict = {"n_seed": len(seeds),
                 "recuperato_frac": rec / max(len(seeds), 1)}
    for k, vs in campioni.items():
        a = np.asarray(vs, dtype=float)
        out[k + "_mean"] = float(a.mean())
        out[k + "_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    if con_adattamento and adatt:
        a = np.asarray(adatt, dtype=float)
        out["adattamento_mean"] = float(a.mean())
        out["adattamento_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return out


def esegui_multiseed(configs: list[dict], T: float, salva_ogni: int,
                     seeds: tuple[int, ...] = SEEDS_DEFAULT,
                     con_adattamento: bool = False) -> list[dict]:
    """Sweep/ablazione multi-seed: una riga per config con media ± std."""
    righe = []
    for cfg in configs:
        p = Param(**cfg["param"])
        stoc = cfg["sim"].get("stocastico", True)
        r = valuta_multiseed(
            p, seeds=seeds, T=T, salva_ogni=salva_ogni,
            rumore_bianco=cfg["sim"].get("rumore_bianco", True),
            protocollo=cfg["sim"].get("protocollo", "stimolo"),
            con_adattamento=con_adattamento,
            stocastico_adatt=stoc, stocastico=stoc)
        r["config"] = cfg["nome"]
        print(f"[{r['config']}] Q={r['qualita_fin_mean']:.4f}±"
              f"{r['qualita_fin_std']:.4f} fed={r['fedelta_mean']:.4f}±"
              f"{r['fedelta_std']:.4f} nov={r['novita_fin_mean']:.5f}±"
              f"{r['novita_fin_std']:.5f} t_rec={r['t_rec_mean']:.3f} "
              f"rec={r['recuperato_frac']:.0%}")
        righe.append(r)
    return righe


def tabella_md_multiseed(righe: list[dict], metriche: list[str]) -> str:
    """Tabella markdown con celle 'media ± std'."""
    head = "| config | " + " | ".join(metriche) + " |"
    sep = "|" + "|".join(["---"] * (len(metriche) + 1)) + "|"
    body = []
    for r in righe:
        celle = [r["config"]]
        for m in metriche:
            mu = r.get(m + "_mean", float("nan"))
            sd = r.get(m + "_std", float("nan"))
            celle.append(f"{mu:.4f} ± {sd:.4f}")
        body.append("| " + " | ".join(celle) + " |")
    return "\n".join([head, sep] + body)


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
               seed: int = 7, seeds: tuple[int, ...] | None = None):
    os.makedirs(out_dir, exist_ok=True)
    configs = config_sweep()
    for c in configs:
        c["param"]["N"] = N
    if seeds:
        return main_sweep_multiseed(N=N, T=T, out_dir=out_dir, seeds=seeds)
    righe = esegui(configs, T=T, salva_ogni=20, seed=seed)
    salva_csv(righe, COLONNE_SWEEP, os.path.join(out_dir, "studio_parametri.csv"))
    md = "# Studio dei parametri\n\n" + tabella_md(righe, COLONNE_SWEEP) + "\n"
    with open(os.path.join(out_dir, "studio_parametri.md"), "w") as f:
        f.write(md)
    print("salvato:", os.path.join(out_dir, "studio_parametri.md"))
    return righe


def main_sweep_multiseed(N: int = 48, T: float = 0.15, out_dir: str = "output",
                         seeds: tuple[int, ...] = SEEDS_DEFAULT):
    """Sweep multi-seed: output studio_parametri_multiseed.csv/.md (media ± std)."""
    os.makedirs(out_dir, exist_ok=True)
    configs = config_sweep()
    for c in configs:
        c["param"]["N"] = N
    righe = esegui_multiseed(configs, T=T, salva_ogni=20, seeds=tuple(seeds),
                             con_adattamento=True)
    metriche = ["qualita_fin", "fedelta", "novita_fin", "adattamento",
                "t_rec"]
    # CSV con mean/std espliciti
    colonne_csv = ["config", "n_seed"] + \
        [m + s for m in (METRICHE_NUMERICHE + ["adattamento"])
         for s in ("_mean", "_std")] + ["recuperato_frac"]
    salva_csv(righe, colonne_csv,
              os.path.join(out_dir, "studio_parametri_multiseed.csv"))
    md = ("# Studio dei parametri (multi-seed, media ± std)\n\n"
          f"Seeds: {list(seeds)}. `t_rec` in finestra T_rec=1.00.\n\n"
          + tabella_md_multiseed(righe, metriche) + "\n")
    with open(os.path.join(out_dir, "studio_parametri_multiseed.md"), "w") as f:
        f.write(md)
    print("salvato:", os.path.join(out_dir, "studio_parametri_multiseed.md"))
    return righe


def main_ablazione(N: int = 48, T: float = 0.15, out_dir: str = "output",
                   seed: int = 7, seeds: tuple[int, ...] | None = None):
    os.makedirs(out_dir, exist_ok=True)
    configs = config_ablazione()
    for c in configs:
        c["param"]["N"] = N
    if seeds:
        return main_ablazione_multiseed(N=N, T=T, out_dir=out_dir,
                                        seeds=seeds)
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


def main_ablazione_multiseed(N: int = 48, T: float = 0.15,
                             out_dir: str = "output",
                             seeds: tuple[int, ...] = SEEDS_DEFAULT):
    """Ablazione multi-seed con adattamento non degenere."""
    os.makedirs(out_dir, exist_ok=True)
    configs = config_ablazione()
    for c in configs:
        c["param"]["N"] = N
    righe = esegui_multiseed(configs, T=T, salva_ogni=20, seeds=tuple(seeds),
                             con_adattamento=True)
    metriche = ["qualita_fin", "fedelta", "novita_fin", "adattamento",
                "t_rec"]
    colonne_csv = ["config", "n_seed"] + \
        [m + s for m in (METRICHE_NUMERICHE + ["adattamento"])
         for s in ("_mean", "_std")] + ["recuperato_frac"]
    salva_csv(righe, colonne_csv,
              os.path.join(out_dir, "ablazione_multiseed.csv"))
    md = ("# Ablazione dei livelli (multi-seed, media ± std)\n\n"
          f"Seeds: {list(seeds)}.\n\n"
          + tabella_md_multiseed(righe, metriche) + "\n")
    with open(os.path.join(out_dir, "ablazione_multiseed.md"), "w") as f:
        f.write(md)
    print("salvato:", os.path.join(out_dir, "ablazione_multiseed.md"))
    # figura con le medie
    righe_medie = []
    for r in righe:
        rm = dict(r)
        for k in ("qualita_fin", "continuita_media", "fedelta",
                  "adattamento"):
            rm[k] = r.get(k + "_mean", 0.0)
        righe_medie.append(rm)
    fig_ablazione(righe_medie, os.path.join(out_dir, "fig08_ablazione.png"))
    return righe
