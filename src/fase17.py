"""Fase 17 — stress fuori regime (solo misura, nessuna modifica al modello).

Cinque assi richiesti dall'utente (25/09/2026), eseguiti senza toccare
equazioni, gf o rete protetta. Questo modulo CHIAMA il core esistente
(`simula`, `verifica_quiete`, `certifica_frammento`, `cue_parziale`,
`ricostruisci_associativo`) e ne misura il comportamento fuori dalla
comfort zone (T brevi, N<=96, bump gaussiani, D standard).

1. TEMPI LUNGHI (T=0.5/1/2/5): dV/dt, drift massa g0, eq/ea, Q, rnov.
2. GRIGLIE ENORMI (N=128/256/512): stabilita' CFL, Turing, gate gf,
   memoria lineare della rete protetta (teorica + sonda su N piccoli).
3. DIFFUSIONE ZERO (D0/Dx/Dy=0 o 0.001): gx senza guida, gy senza
   diffusione armonica, gf su stati spigolosi.
4. CUE NON STRUTTURATI (multi-bump, sparsi, random, striscia, checker):
   adattamento, gate, certificazione, Q, associativo.
5. DATI REALI / NATURALI (gradiente, barre, chirp, ripple, multiscala,
   eeg-sintetico, normalizzati a massa 1): certificazione, ricostruzione,
   retrieval nearest-MSE, fedelta', qmask.

Solo numpy + matplotlib (figure) + std. Tutto deterministico
(stocastico=False) per isolare la dinamica dal rumore, gia'
caratterizzato in §6. Vincoli onesti documentati in ogni report.

Nota dt: `Param.dt_stabile()` divide per max(D); con D=0 non e'
definito e con D piccolissimi da' dt enormi (inaccurati per la parte
reattiva). `_dt_sicuro` usa dt_stabile quando sensato e un fallback
reattivo (5e-4 / cap 2e-3) altrimenti. Non modifica `frammento_2d`.
"""

from __future__ import annotations

import csv as _csv
import os as _os
import time as _time

import numpy as np

from .frammento_2d import Param, essenza, simula, laplaciano, turing_gy
from .frammento_gf import verifica_quiete, certifica_frammento
from .rete_frammento import (
    cue_parziale,
    ricostruisci_associativo,
    _q_regione,
    genera_cue,
    ReteFrammento,
)

OUT_DIR_DEFAULT = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "output", "output_test")

DT_FALLBACK_ZERO = 5e-4  # D tutte nulle: solo reazione, Eulero esplicito
DT_CAP = 2e-3            # mai passi piu' larghi (accuratezza reattiva)

# eq = err_fx_fo (operativita' vs Fo, soglia eps) ;
# ea = err_fo_ess (Fo vs essenza, soglia delta). Nomi dal diario Fase 9.


# ===========================================================================
# 0. Utilita' condivise
# ===========================================================================
def dt_sicuro(p: Param) -> float:
    """Passo temporale per le misure: dt_stabile se sensato, fallback se no."""
    dmax = max(float(p.D0), float(p.Dx), float(p.Dy))
    if dmax <= 1e-12:
        return float(DT_FALLBACK_ZERO)
    dt = float(p.dt_stabile())
    return float(min(dt, DT_CAP))


def salva_ogni_per(T: float, dt: float, max_snap: int = 120) -> int:
    """Salva ~max_snap snapshot qualunque sia T (memoria limitata)."""
    nsteps = max(1, int(float(T) / float(dt)))
    return max(1, nsteps // int(max_snap))


def misura_snap(snap: dict, p: Param) -> dict:
    """Metriche standard da uno snapshot: dV, masse, eq/ea, Q, rnov, cert."""
    from .frammento_2d import derivata_numerica

    d = snap["diag"]
    dV = derivata_numerica(np.asarray(d["t"]), np.asarray(d["V"]))
    fraz_dv = float(np.mean(np.asarray(dV) <= 1e-9)) if dV.size else 1.0
    drift_massa = float(d["massa0"][-1] - d["massa0"][0])
    Fo, Fx, Fy = snap["F0"][-1], snap["Fx"][-1], snap["Fy"][-1]
    ess = snap["essenza"]
    q = verifica_quiete(Fo, Fx, ess, p.dx, Fy=Fy, V_hist=d["V"])
    c = certifica_frammento(Fo, Fx, Fy, ess, p.dx, diag=dict(d))
    n_ess = float(np.linalg.norm(np.asarray(ess)))
    nov_ass = float((np.asarray(Fy) * 1.0).sum() * p.dx ** 2)
    finiti = bool(np.all(np.isfinite(Fo)) and np.all(np.isfinite(Fx))
                  and np.all(np.isfinite(Fy)))
    return {
        "fraz_dV_nonpos": fraz_dv,
        "V0": float(d["V"][0]),
        "Vfine": float(d["V"][-1]),
        "massa0_in": float(d["massa0"][0]),
        "massa0_fin": float(d["massa0"][-1]),
        "drift_massa_g0": drift_massa,
        "eq": float(q["err_fx_fo"]),
        "ea": float(q["err_fo_ess"]),
        "quiete": bool(q["attivo"]),
        "Q": float(c["qualita"]),
        "rnov": float(c["novita_rel"]),
        "nov_ass": nov_ass,
        "cert": bool(c["certificato"]),
        "gate_medio": float(d["gate_medio"][-1]),
        "n_ess": n_ess,
        "finiti": finiti,
        "motivo": str(c["motivo"]),
    }


def _mesh(N: int, L: float = 1.0):
    x = (np.arange(int(N)) + 0.5) / float(N) * float(L)
    return np.meshgrid(x, x, indexing="ij")


# ===========================================================================
# 1. TEMPI LUNGHI
# ===========================================================================
def valuta_tempi_lunghi(N: int = 48, T_list=(0.5, 1.0, 2.0, 5.0),
                        seed: int = 7,
                        protocollo: str = "stimolo") -> list[dict]:
    """Una run deterministica per ogni T: stabilita' fuori regime breve."""
    righe: list[dict] = []
    for T in T_list:
        p = Param(N=int(N), seed=int(seed))
        dt = dt_sicuro(p)
        so = salva_ogni_per(float(T), dt)
        t0 = _time.perf_counter()
        snap = simula(p, T=float(T), dt=dt, stocastico=False,
                      protocollo=protocollo, salva_ogni=so)
        wall = _time.perf_counter() - t0
        m = misura_snap(snap, p)
        righe.append({"N": int(N), "T": float(T), "dt": float(dt),
                      "nsteps": int(float(T) / dt),
                      "wall_s": float(wall), **m})
    return righe


def salva_report_tempi(righe: list[dict], out_dir: str | None = None,
                       N: int = 48, seed: int = 7) -> dict:
    """CSV + md + fig24 (Q/eq-ea/drift/V vs T)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = OUT_DIR_DEFAULT
    _os.makedirs(out_dir, exist_ok=True)
    fp_csv = _os.path.join(out_dir, "fase17_tempi.csv")
    with open(fp_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["N", "T", "dt", "nsteps", "wall_s", "fraz_dV",
                    "V0", "Vfine", "drift_massa_g0", "eq", "ea",
                    "quiete", "Q", "rnov", "cert", "finiti"])
        for r in righe:
            w.writerow([r["N"], f"{r['T']:.2f}", f"{r['dt']:.2e}",
                        r["nsteps"], f"{r['wall_s']:.2f}",
                        f"{r['fraz_dV_nonpos']:.4f}",
                        f"{r['V0']:.6f}", f"{r['Vfine']:.6f}",
                        f"{r['drift_massa_g0']:.2e}",
                        f"{r['eq']:.4f}", f"{r['ea']:.4f}",
                        int(r["quiete"]), f"{r['Q']:.4f}",
                        f"{r['rnov']:.4f}", int(r["cert"]),
                        int(r["finiti"])])
    fp_md = _os.path.join(out_dir, "fase17_tempi.md")
    with open(fp_md, "w") as f:
        f.write("# Fase 17.1 — Tempi lunghi (deterministico, N=%d seed=%d)\n\n" % (N, seed))
        f.write("Una run per T. Atteso: dV/dt<=0 resta ~1.0 (strutturale), "
                "Fo diffonde verso il piatto quindi ea cresce e la quiete "
                "tarata sul regime breve puo' cadere a T grandi "
                "(risultato onesto, non bug).\n\n")
        f.write("| T | dV<=0 | V0->Vfine | drift g0 | eq | ea | Q | rnov | quiete | cert |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in righe:
            f.write(f"| {r['T']:.2f} | {r['fraz_dV_nonpos']:.3f} | "
                    f"{r['V0']:.3f}->{r['Vfine']:.3f} | "
                    f"{r['drift_massa_g0']:.1e} | {r['eq']:.3f} | "
                    f"{r['ea']:.3f} | {r['Q']:.3f} | {r['rnov']:.3f} | "
                    f"{'SI' if r['quiete'] else 'NO'} | "
                    f"{'SI' if r['cert'] else 'NO'} |\n")
        f.write("\nLimite: seed singolo, protocollo stimolo; lo stimolo "
                "Lissajous ha periodi 0.30/0.45 quindi a T>>0.45 la media "
                "e' su molti cicli.\n")
    Ts = [r["T"] for r in righe]
    fig, ax = plt.subplots(2, 2, figsize=(12.0, 8.0))
    ax[0, 0].plot(Ts, [r["Q"] for r in righe], "o-")
    ax[0, 0].axhline(0.40, color="r", ls="--", label="soglia Q")
    ax[0, 0].set_xlabel("T")
    ax[0, 0].set_ylabel("Q")
    ax[0, 0].set_title("A) Qualita' vs orizzonte")
    ax[0, 0].legend()
    ax[0, 1].plot(Ts, [r["eq"] for r in righe], "o-", label="eq (eps=0.60)")
    ax[0, 1].plot(Ts, [r["ea"] for r in righe], "s-", label="ea (delta=0.90)")
    ax[0, 1].axhline(0.60, color="r", ls="--")
    ax[0, 1].axhline(0.90, color="r", ls=":")
    ax[0, 1].set_xlabel("T")
    ax[0, 1].set_title("B) Quiete vs orizzonte")
    ax[0, 1].legend()
    ax[1, 0].plot(Ts, [r["V0"] for r in righe], "o-", label="V0")
    ax[1, 0].plot(Ts, [r["Vfine"] for r in righe], "s-", label="Vfine")
    ax[1, 0].set_xlabel("T")
    ax[1, 0].set_title("C) Lyapunov iniziale/finale")
    ax[1, 0].legend()
    ax[1, 1].plot(Ts, [r["drift_massa_g0"] for r in righe], "o-")
    ax[1, 1].set_xlabel("T")
    ax[1, 1].set_title("D) Drift massa g0 (deve restare ~1e-12)")
    fig.suptitle(f"Fase 17.1 — tempi lunghi (N={N}, det.)")
    fig.tight_layout()
    fp_fig = _os.path.join(out_dir, "fig24_tempi.png")
    fig.savefig(fp_fig, dpi=110)
    plt.close(fig)
    return {"csv": fp_csv, "md": fp_md, "fig": fp_fig}


# ===========================================================================
# 2. GRIGLIE ENORMI
# ===========================================================================
def valuta_griglie(N_list=(32, 64, 128, 256), T: float = 0.05,
                   seed: int = 7, sonda_rete: bool = True) -> list[dict]:
    """Una run per N + fumo Turing (solo N<=64) + memoria teorica/sonda."""
    righe: list[dict] = []
    for N in N_list:
        N = int(N)
        p = Param(N=N, seed=int(seed))
        dt = dt_sicuro(p)
        so = salva_ogni_per(float(T), dt)
        t0 = _time.perf_counter()
        snap = simula(p, T=float(T), dt=dt, stocastico=False,
                      protocollo="stimolo", salva_ogni=so)
        wall = _time.perf_counter() - t0
        m = misura_snap(snap, p)
        # Turing: dt fisso 2.5e-4 instabile oltre N~64 -> solo N piccoli.
        if N <= 64:
            tg = turing_gy(p, passi=2000)
            t_std = float(np.asarray(tg["a"]).std())
            t_max = float(np.asarray(tg["a"]).max())
            turing_nota = "ok"
        else:
            t_std, t_max = float("nan"), float("nan")
            turing_nota = "non valutato (dt Turing fisso instabile oltre N=64)"
        mem_teorica = int(N * N * 8 * 4)  # Fx,Fo,Fy,ess float64
        n_slot = None
        tasso_cert = None
        if sonda_rete and N <= 128:
            rete = ReteFrammento(N=N, T=min(float(T), 0.05))
            cue = genera_cue(3, seed=int(seed))
            ok = 0
            for c in cue:
                r = rete.impara(c)
                ok += int(bool(r["cert"]))
            n_slot = len(rete.slot)
            tasso_cert = ok / 3.0
        righe.append({"N": N, "T": float(T), "dt": float(dt),
                      "nsteps": int(float(T) / dt), "wall_s": float(wall),
                      "turing_std": t_std, "turing_max": t_max,
                      "turing_nota": turing_nota,
                      "mem_slot_bytes": mem_teorica,
                      "sonda_slot": n_slot, "sonda_tasso_cert": tasso_cert,
                      **m})
    return righe


def salva_report_griglie(righe: list[dict], out_dir: str | None = None,
                         T: float = 0.05, seed: int = 7) -> dict:
    """CSV + md + fig25 (wall log, Q, memoria)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = OUT_DIR_DEFAULT
    _os.makedirs(out_dir, exist_ok=True)
    fp_csv = _os.path.join(out_dir, "fase17_griglie.csv")
    with open(fp_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["N", "T", "dt", "nsteps", "wall_s", "fraz_dV",
                    "drift_massa_g0", "eq", "ea", "Q", "cert",
                    "turing_std", "mem_slot_bytes", "sonda_tasso_cert",
                    "finiti"])
        for r in righe:
            w.writerow([r["N"], f"{r['T']:.3f}", f"{r['dt']:.2e}",
                        r["nsteps"], f"{r['wall_s']:.2f}",
                        f"{r['fraz_dV_nonpos']:.4f}",
                        f"{r['drift_massa_g0']:.2e}",
                        f"{r['eq']:.4f}", f"{r['ea']:.4f}",
                        f"{r['Q']:.4f}", int(r["cert"]),
                        f"{r['turing_std']:.4f}" if r["turing_std"] == r["turing_std"] else "n/d",
                        r["mem_slot_bytes"],
                        (f"{r['sonda_tasso_cert']:.2f}"
                         if r["sonda_tasso_cert"] is not None else "n/d"),
                        int(r["finiti"])])
    fp_md = _os.path.join(out_dir, "fase17_griglie.md")
    with open(fp_md, "w") as f:
        f.write("# Fase 17.2 — Griglie enormi (T=%.3f det., seed=%d)\n\n" % (T, seed))
        f.write("Costo ~N^4 a T fisso (passi ~N^2, costo/passo ~N^2). "
                "Turing solo N<=64 (dt fisso instabile oltre). "
                "Sonda rete (3 cue) solo N<=128.\n\n")
        f.write("| N | dt | passi | wall s | dV<=0 | drift g0 | Q | cert | mem/slot | sonda |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in righe:
            memkb = r["mem_slot_bytes"] / 1024.0
            sonda = ("n/d" if r["sonda_tasso_cert"] is None
                     else f"{r['sonda_tasso_cert']:.2f}")
            f.write(f"| {r['N']} | {r['dt']:.1e} | {r['nsteps']} | "
                    f"{r['wall_s']:.1f} | {r['fraz_dV_nonpos']:.3f} | "
                    f"{r['drift_massa_g0']:.1e} | {r['Q']:.3f} | "
                    f"{'SI' if r['cert'] else 'NO'} | {memkb:.0f} KB | "
                    f"{sonda} |\n")
        f.write("\nLimite: seed singolo; N=512 solo su richiesta "
                "(--include-512) per costo.\n")
    Ns = [r["N"] for r in righe]
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.5))
    ax[0].loglog(Ns, [max(r["wall_s"], 1e-3) for r in righe], "o-")
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("wall s (log)")
    ax[0].set_title("A) Costo vs N (~N^4 atteso)")
    ax[1].plot(Ns, [r["Q"] for r in righe], "o-")
    ax[1].axhline(0.40, color="r", ls="--")
    ax[1].set_xlabel("N")
    ax[1].set_ylabel("Q")
    ax[1].set_title("B) Qualita' vs N")
    ax[2].loglog(Ns, [r["mem_slot_bytes"] for r in righe], "o-")
    ax[2].set_xlabel("N")
    ax[2].set_ylabel("byte/slot (log)")
    ax[2].set_title("C) Memoria lineare teorica")
    fig.suptitle(f"Fase 17.2 — griglie (T={T}, det.)")
    fig.tight_layout()
    fp_fig = _os.path.join(out_dir, "fig25_griglie.png")
    fig.savefig(fp_fig, dpi=110)
    plt.close(fig)
    return {"csv": fp_csv, "md": fp_md, "fig": fp_fig}


# ===========================================================================
# 3. DIFFUSIONE ZERO
# ===========================================================================
CONFIG_DIFFUSIONE = (
    ("base", None),
    ("D0=0", {"D0": 0.0}),
    ("Dx=0", {"Dx": 0.0}),
    ("Dy=0", {"Dy": 0.0}),
    ("tutto-zero", {"D0": 0.0, "Dx": 0.0, "Dy": 0.0}),
    ("quasi-zero", {"D0": 0.001, "Dx": 0.001, "Dy": 0.001}),
)


def valuta_diffusione(N: int = 32, T: float = 0.10, seed: int = 7,
                      configs=CONFIG_DIFFUSIONE) -> list[dict]:
    """Stessa run con diffusioni azzerate/ridotte: la diffusione e' essenziale?"""
    righe: list[dict] = []
    for nome, mod in configs:
        kw: dict = {}
        if mod:
            kw.update(mod)
        p = Param(N=int(N), seed=int(seed), **kw)
        dt = dt_sicuro(p)
        so = salva_ogni_per(float(T), dt)
        snap = simula(p, T=float(T), dt=dt, stocastico=False,
                      protocollo="stimolo", salva_ogni=so)
        m = misura_snap(snap, p)
        Fx = np.asarray(snap["Fx"][-1])
        ruvido = float(np.mean(np.abs(laplaciano(Fx, p.dx))))
        righe.append({"config": nome, "N": int(N), "T": float(T),
                      "D0": float(p.D0), "Dx": float(p.Dx),
                      "Dy": float(p.Dy), "dt": float(dt),
                      "ruvidezza": ruvido, **m})
    return righe


def salva_report_diffusione(righe: list[dict], out_dir: str | None = None,
                            N: int = 32, T: float = 0.10,
                            seed: int = 7) -> dict:
    """CSV + md + fig26 (Q + ruvidezza per config)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = OUT_DIR_DEFAULT
    _os.makedirs(out_dir, exist_ok=True)
    fp_csv = _os.path.join(out_dir, "fase17_diffusione.csv")
    with open(fp_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["config", "D0", "Dx", "Dy", "dt", "fraz_dV", "Q",
                    "eq", "ea", "rnov", "cert", "ruvidezza", "finiti"])
        for r in righe:
            w.writerow([r["config"], f"{r['D0']:.4f}", f"{r['Dx']:.4f}",
                        f"{r['Dy']:.4f}", f"{r['dt']:.2e}",
                        f"{r['fraz_dV_nonpos']:.4f}", f"{r['Q']:.4f}",
                        f"{r['eq']:.4f}", f"{r['ea']:.4f}",
                        f"{r['rnov']:.4f}", int(r["cert"]),
                        f"{r['ruvidezza']:.4f}", int(r["finiti"])])
    fp_md = _os.path.join(out_dir, "fase17_diffusione.md")
    with open(fp_md, "w") as f:
        f.write("# Fase 17.3 — Diffusione zero/ridotta (N=%d T=%.2f seed=%d)\n\n"
                % (N, T, seed))
        f.write("`ruvidezza` = media |lap(Fx)| finale (stati spigolosi se "
                "alta). Con D=0 il dt e' fallback reattivo 5e-4 "
                "(dt_stabile non definito); vedi `_dt_sicuro`.\n\n")
        f.write("| config | D0/Dx/Dy | Q | eq | rnov | cert | ruvidezza |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in righe:
            f.write(f"| {r['config']} | {r['D0']:.3f}/{r['Dx']:.3f}/{r['Dy']:.3f} | "
                    f"{r['Q']:.3f} | {r['eq']:.3f} | {r['rnov']:.3f} | "
                    f"{'SI' if r['cert'] else 'NO'} | {r['ruvidezza']:.2f} |\n")
        f.write("\nMotivi di mancata certificazione:\n\n")
        for r in righe:
            if not r["cert"]:
                f.write(f"- {r['config']}: {r['motivo']}\n")
        f.write("\nLettura: se Q/cert crollano senza diffusione, la "
                "diffusione e' essenziale; se tengono, e' solo conveniente.\n")
    nomi = [r["config"] for r in righe]
    xs = np.arange(len(nomi))
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.5))
    ax[0].bar(xs, [r["Q"] for r in righe])
    ax[0].axhline(0.40, color="r", ls="--")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(nomi, rotation=20, ha="right")
    ax[0].set_ylabel("Q")
    ax[0].set_title("A) Qualita' per config")
    ax[1].bar(xs, [r["ruvidezza"] for r in righe])
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(nomi, rotation=20, ha="right")
    ax[1].set_ylabel("ruvidezza")
    ax[1].set_title("B) Spigolosita' per config")
    fig.suptitle(f"Fase 17.3 — diffusione (N={N}, det.)")
    fig.tight_layout()
    fp_fig = _os.path.join(out_dir, "fig26_diffusione.png")
    fig.savefig(fp_fig, dpi=110)
    plt.close(fig)
    return {"csv": fp_csv, "md": fp_md, "fig": fp_fig}


# ===========================================================================
# 4. CUE NON STRUTTURATI
# ===========================================================================
TIPI_CUE = ("multi-bump", "sparso", "random", "striscia", "checker")


def _pattern_cue(tipo: str, N: int, rng: np.random.Generator,
                 ess: np.ndarray) -> np.ndarray:
    """Fx0 = ess + pattern non gaussiano (massa extra comparabile al bump)."""
    X, Y = _mesh(N)
    L = 1.0
    m_ess = float(np.asarray(ess).mean())
    if tipo == "multi-bump":
        F = np.asarray(ess).copy()
        for _ in range(3):
            cx, cy = float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))
            amp = float(rng.uniform(1.0, 3.0))
            F = F + amp * np.exp(-((X - cx * L) ** 2 + (Y - cy * L) ** 2)
                                 / (2 * 0.05 ** 2))
        return F
    if tipo == "sparso":
        F = np.asarray(ess).copy()
        idx = rng.random((N, N)) < 0.02
        F[idx] += float(rng.uniform(2.0, 4.0)) * m_ess * 10.0
        return F
    if tipo == "random":
        F = (np.asarray(ess)
             + 2.0 * (rng.random((N, N)) - 0.5) * m_ess * 2.0)
        return np.maximum(F, 0.0)
    if tipo == "striscia":
        F = np.asarray(ess).copy()
        x0 = float(rng.uniform(0.2, 0.7))
        banda = (X >= x0 * L) & (X < (x0 + 0.10) * L)
        F[banda] += 2.0 * m_ess * 10.0
        return F
    if tipo == "checker":
        onda = np.sin(4 * np.pi * X / L) * np.sin(4 * np.pi * Y / L)
        F = np.asarray(ess) + 2.0 * (0.5 + 0.5 * onda) * m_ess * 2.0
        return np.maximum(F, 0.0)
    raise ValueError(f"tipo cue sconosciuto: {tipo!r}")


def valuta_cue_nonstrutturati(N: int = 32, T: float = 0.10, seed: int = 7,
                              tipi=TIPI_CUE, n_per_tipo: int = 8) -> list[dict]:
    """Per tipo: tasso cert, Q, eq, gate, rnov + sonda associativa."""
    righe: list[dict] = []
    for tipo in tipi:
        Qs, eqs, gates, rnovs = [], [], [], []
        ncert = 0
        certificati: list[dict] = []
        for k in range(int(n_per_tipo)):
            rng = np.random.default_rng(int(seed) * 7919 + hash(tipo) % 10_000 + k)
            p = Param(N=int(N), seed=int(seed) * 1000 + k)
            ess = essenza(p)
            Fx0 = _pattern_cue(tipo, int(N), rng, ess)
            dt = dt_sicuro(p)
            so = salva_ogni_per(float(T), dt)
            snap = simula(p, T=float(T), dt=dt, stocastico=False,
                          protocollo="stimolo", salva_ogni=so,
                          stato_iniziale={"Fx": Fx0})
            m = misura_snap(snap, p)
            Qs.append(m["Q"])
            eqs.append(m["eq"])
            gates.append(m["gate_medio"])
            rnovs.append(m["rnov"])
            ncert += int(m["cert"])
            if m["cert"]:
                certificati.append({"Fx": np.asarray(snap["Fx"][-1]),
                                    "Fo": np.asarray(snap["F0"][-1])})
        # sonda associativa sui primi 2 certificati (se esistono)
        qmasks: list[float] = []
        dx = 1.0 / float(N)
        for rec in certificati[:2]:
            cp, vis = cue_parziale(rec["Fx"], frazione=0.5, tipo="blocco",
                                   rumore=0.0, seed=int(seed))
            Fr = ricostruisci_associativo(cp, vis, rec["Fo"], dx,
                                          modo="residuo")
            qmasks.append(_q_regione(Fr, rec["Fx"], ~vis))
        righe.append({"tipo": tipo, "n": int(n_per_tipo),
                      "tasso_cert": ncert / max(1, int(n_per_tipo)),
                      "Q_media": float(np.mean(Qs)) if Qs else 0.0,
                      "Q_min": float(np.min(Qs)) if Qs else 0.0,
                      "eq_media": float(np.mean(eqs)) if eqs else 0.0,
                      "gate_medio": float(np.mean(gates)) if gates else 0.0,
                      "rnov_media": float(np.mean(rnovs)) if rnovs else 0.0,
                      "qmask_media": (float(np.mean(qmasks))
                                      if qmasks else float("nan")),
                      "n_qmask": len(qmasks)})
    return righe


def salva_report_cue(righe: list[dict], out_dir: str | None = None,
                     N: int = 32, T: float = 0.10, seed: int = 7) -> dict:
    """CSV + md + fig27 (tasso cert + Q per tipo)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = OUT_DIR_DEFAULT
    _os.makedirs(out_dir, exist_ok=True)
    fp_csv = _os.path.join(out_dir, "fase17_cue.csv")
    with open(fp_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["tipo", "n", "tasso_cert", "Q_media", "Q_min",
                    "eq_media", "gate_medio", "rnov_media", "qmask_media"])
        for r in righe:
            w.writerow([r["tipo"], r["n"], f"{r['tasso_cert']:.3f}",
                        f"{r['Q_media']:.4f}", f"{r['Q_min']:.4f}",
                        f"{r['eq_media']:.4f}", f"{r['gate_medio']:.4f}",
                        f"{r['rnov_media']:.4f}",
                        (f"{r['qmask_media']:.4f}"
                         if r["qmask_media"] == r["qmask_media"] else "n/d")])
    fp_md = _os.path.join(out_dir, "fase17_cue.md")
    with open(fp_md, "w") as f:
        f.write("# Fase 17.4 — Cue non strutturati (N=%d T=%.2f seed=%d)\n\n"
                % (N, T, seed))
        f.write("Fx0 = essenza + pattern (energia extra comparabile al bump "
                "gaussiano). Sonda associativa: blocco 0.5, residuo, sui "
                "primi 2 certificati per tipo (n/d se nessun certificato).\n\n")
        f.write("| tipo | tasso cert | Q media (min) | eq | gate | rnov | qmask |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in righe:
            qm = ("n/d" if r["qmask_media"] != r["qmask_media"]
                  else f"{r['qmask_media']:.3f}")
            f.write(f"| {r['tipo']} | {r['tasso_cert']:.2f} | "
                    f"{r['Q_media']:.3f} ({r['Q_min']:.3f}) | "
                    f"{r['eq_media']:.3f} | {r['gate_medio']:.3f} | "
                    f"{r['rnov_media']:.3f} | {qm} |\n")
        f.write("\nLimite: seed singolo; pattern sintetici (non dati veri, "
                "quelli sono nell'asse 5).\n")
    nomi = [r["tipo"] for r in righe]
    xs = np.arange(len(nomi))
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.5))
    ax[0].bar(xs, [r["tasso_cert"] for r in righe])
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(nomi, rotation=20, ha="right")
    ax[0].set_ylabel("tasso certificazione")
    ax[0].set_title("A) Chi passa il gate gf")
    ax[1].bar(xs, [r["Q_media"] for r in righe])
    ax[1].axhline(0.40, color="r", ls="--")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(nomi, rotation=20, ha="right")
    ax[1].set_ylabel("Q media")
    ax[1].set_title("B) Qualita' per tipo")
    fig.suptitle(f"Fase 17.4 — cue non strutturati (N={N}, det.)")
    fig.tight_layout()
    fp_fig = _os.path.join(out_dir, "fig27_cue.png")
    fig.savefig(fp_fig, dpi=110)
    plt.close(fig)
    return {"csv": fp_csv, "md": fp_md, "fig": fp_fig}


# ===========================================================================
# 5. DATI REALI / NATURALI (stand-in sintetici + file opzionali)
# ===========================================================================
CAMP_REALI = ("gradiente", "barre", "chirp", "ripple", "multiscala",
              "eeg-sintetico")


def campo_naturale(nome: str, N: int, seed: int = 7) -> np.ndarray:
    """Campo dimostrativo normalizzato dopo (massa 1, >=0)."""
    X, Y = _mesh(N)
    L = 1.0
    rng = np.random.default_rng(int(seed) + hash(nome) % 1000)
    if nome == "gradiente":
        F = 1.0 + X / L + 0.5 * Y / L
    elif nome == "barre":
        F = 1.0 + (np.sign(np.sin(8 * np.pi * X / L)) > 0).astype(float)
    elif nome == "chirp":
        tt = X[:, 0]
        chirp = np.sin(2 * np.pi * (2.0 * tt + 8.0 * tt ** 2))
        F = np.tile((1.5 + chirp)[:, None], (1, N))
    elif nome == "ripple":
        F = (1.5 + np.sin(6 * np.pi * X / L) * np.cos(5 * np.pi * Y / L)
             + 0.3 * rng.random((N, N)))
    elif nome == "multiscala":
        F = np.ones((N, N))
        for sig, amp in ((0.02, 3.0), (0.05, 2.0), (0.10, 1.0)):
            cx, cy = float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.3, 0.7))
            F = F + amp * np.exp(-((X - cx * L) ** 2 + (Y - cy * L) ** 2)
                                 / (2 * sig ** 2))
    elif nome == "eeg-sintetico":
        t = np.linspace(0, 2, N)
        theta = np.sin(2 * np.pi * 6.0 * t / 2.0)
        gamma = np.sin(2 * np.pi * 45.0 * t / 2.0) * (0.5 + 0.5 * theta)
        traccia = 2.0 + theta + 0.5 * gamma
        F = np.tile(traccia[:, None], (1, N))
    else:
        raise ValueError(f"campo sconosciuto: {nome!r}")
    F = np.maximum(np.asarray(F, dtype=float), 0.0)
    dx = 1.0 / float(N)
    massa = float(F.sum() * dx ** 2)
    return F / max(massa, 1e-12)


def valuta_reali(N: int = 32, T: float = 0.10, seed: int = 7,
                 nomi=CAMP_REALI) -> list[dict]:
    """Certificazione + retrieval nearest-MSE + qmask per campo naturale."""
    # 1. certifica ogni campo come Fx0
    certificati: list[dict] = []
    info: dict[str, dict] = {}
    for i, nome in enumerate(nomi):
        F_vero = campo_naturale(nome, int(N), seed=int(seed))
        p = Param(N=int(N), seed=int(seed) * 1000 + i)
        ess = essenza(p)
        dt = dt_sicuro(p)
        so = salva_ogni_per(float(T), dt)
        snap = simula(p, T=float(T), dt=dt, stocastico=False,
                      protocollo="stimolo", salva_ogni=so,
                      stato_iniziale={"Fx": F_vero})
        m = misura_snap(snap, p)
        info[nome] = {"m": m, "F_vero": F_vero,
                      "Fx": np.asarray(snap["Fx"][-1]),
                      "Fo": np.asarray(snap["F0"][-1])}
        if m["cert"]:
            certificati.append({"nome": nome, "Fx": info[nome]["Fx"],
                                "Fo": info[nome]["Fo"],
                                "F_vero": F_vero})
    # 2. retrieval: cue parziale del vero certificato, scelta nearest-MSE
    dx = 1.0 / float(N)
    righe: list[dict] = []
    for rum in (0.0, 0.5):
        ok = 0
        qms: list[float] = []
        for rec in certificati:
            cp, vis = cue_parziale(rec["F_vero"], frazione=0.5,
                                   tipo="blocco", rumore=float(rum),
                                   seed=hash(rec["nome"]) % 10_000)
            best, best_mse = None, float("inf")
            for cand in certificati:
                d = cp[vis] - cand["F_vero"][vis]
                mse = float(np.mean(d ** 2))
                if mse < best_mse:
                    best_mse, best = mse, cand["nome"]
            ok += int(best == rec["nome"])
            prior = next(c["Fo"] for c in certificati if c["nome"] == best)
            Fr = ricostruisci_associativo(cp, vis, prior, dx, modo="residuo")
            qms.append(_q_regione(Fr, rec["F_vero"], ~vis))
        n = max(1, len(certificati))
        righe.append({"rumore": float(rum), "n_cert": len(certificati),
                      "n_tot": len(nomi),
                      "acc": (ok / n) if certificati else 0.0,
                      "qmask_media": float(np.mean(qms)) if qms else 0.0,
                      "dettagli": {k: v["m"] for k, v in info.items()}})
    return righe


def salva_report_reali(righe: list[dict], out_dir: str | None = None,
                       N: int = 32, T: float = 0.10, seed: int = 7) -> dict:
    """CSV + md + fig28 (cert per campo + retrieval)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = OUT_DIR_DEFAULT
    _os.makedirs(out_dir, exist_ok=True)
    dett = righe[0]["dettagli"] if righe else {}
    fp_csv = _os.path.join(out_dir, "fase17_reali.csv")
    with open(fp_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["sezione", "campo_rumore", "tasso_cert_acc",
                    "Q", "eq", "rnov", "qmask"])
        for nome, m in dett.items():
            w.writerow(["cert", nome, int(m["cert"]), f"{m['Q']:.4f}",
                        f"{m['eq']:.4f}", f"{m['rnov']:.4f}", ""])
        for r in righe:
            w.writerow(["retrieval", f"{r['rumore']:.1f}",
                        f"{r['acc']:.3f}", "", "", "",
                        f"{r['qmask_media']:.4f}"])
    fp_md = _os.path.join(out_dir, "fase17_reali.md")
    with open(fp_md, "w") as f:
        f.write("# Fase 17.5 — Dati naturali (N=%d T=%.2f seed=%d)\n\n"
                % (N, T, seed))
        f.write("Campi dimostrativi NON gaussiani, normalizzati a massa 1. "
                "Sono stand-in onesti (non claim di validazione reale): "
                "il protocollo accetta anche PNG/WAV/CSV veri convertiti "
                "allo stesso formato (array >=0, massa 1, lato N).\n\n")
        f.write("| campo | cert | Q | eq | rnov |\n")
        f.write("|---|---|---|---|---|\n")
        for nome, m in dett.items():
            f.write(f"| {nome} | {'SI' if m['cert'] else 'NO'} | "
                    f"{m['Q']:.3f} | {m['eq']:.3f} | {m['rnov']:.3f} |\n")
        f.write("\nMotivi di mancata certificazione:\n\n")
        for nome, m in dett.items():
            if not m["cert"]:
                f.write(f"- {nome}: {m['motivo']}\n")
        f.write("\n## Retrieval nearest-MSE sui certificati\n\n")
        f.write("| rumore | n_cert/n_tot | acc | qmask |\n")
        f.write("|---|---|---|---|\n")
        for r in righe:
            f.write(f"| {r['rumore']:.1f} | {r['n_cert']}/{r['n_tot']} | "
                    f"{r['acc']:.3f} | {r['qmask_media']:.3f} |\n")
        f.write("\nLimite: mapping immagine/audio->campo e' arbitrario; "
                "EEG veri restano il pilota nullo di Fase 12.\n")
    nomi = list(dett.keys())
    xs = np.arange(len(nomi))
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.5))
    ax[0].bar(xs, [dett[k]["Q"] for k in nomi])
    ax[0].axhline(0.40, color="r", ls="--")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(nomi, rotation=20, ha="right")
    ax[0].set_ylabel("Q")
    ax[0].set_title("A) Qualita' per campo")
    rr = [r["rumore"] for r in righe]
    ax[1].plot(rr, [r["acc"] for r in righe], "o-", label="acc")
    ax[1].plot(rr, [r["qmask_media"] for r in righe], "s-", label="qmask")
    ax[1].set_xlabel("rumore")
    ax[1].set_title("B) Retrieval sui certificati")
    ax[1].legend()
    fig.suptitle(f"Fase 17.5 — naturali (N={N}, det.)")
    fig.tight_layout()
    fp_fig = _os.path.join(out_dir, "fig28_reali.png")
    fig.savefig(fp_fig, dpi=110)
    plt.close(fig)
    return {"csv": fp_csv, "md": fp_md, "fig": fp_fig}


# ===========================================================================
# Runner completo
# ===========================================================================
def esegui_tutti(N_tempi: int = 48, T_griglie: float = 0.05,
                 N_cue_reali: int = 32, T_altri: float = 0.10,
                 seed: int = 7, out_dir: str | None = None,
                 include_512: bool = False) -> dict:
    """Esegue i 5 assi e salva tutti i report. Ritorna i path."""
    out = {}
    r1 = valuta_tempi_lunghi(N=N_tempi, seed=seed)
    out["tempi"] = salva_report_tempi(r1, out_dir=out_dir, N=N_tempi,
                                      seed=seed)
    nl = [32, 64, 128, 256] + ([512] if include_512 else [])
    r2 = valuta_griglie(N_list=nl, T=T_griglie, seed=seed)
    out["griglie"] = salva_report_griglie(r2, out_dir=out_dir,
                                          T=T_griglie, seed=seed)
    r3 = valuta_diffusione(N=N_cue_reali, T=T_altri, seed=seed)
    out["diffusione"] = salva_report_diffusione(r3, out_dir=out_dir,
                                                N=N_cue_reali, T=T_altri,
                                                seed=seed)
    r4 = valuta_cue_nonstrutturati(N=N_cue_reali, T=T_altri, seed=seed)
    out["cue"] = salva_report_cue(r4, out_dir=out_dir, N=N_cue_reali,
                                  T=T_altri, seed=seed)
    r5 = valuta_reali(N=N_cue_reali, T=T_altri, seed=seed)
    out["reali"] = salva_report_reali(r5, out_dir=out_dir,
                                      N=N_cue_reali, T=T_altri, seed=seed)
    return out
