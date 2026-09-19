"""Frammento del veloce - stress test 500 domande (g0 -> gf).

Ogni "domanda" e' un cue di richiamo: bump gaussiano in posizione/ampiezza
casuale sovrapposto all'essenza come `stato_iniziale` di `Fx`, con seed,
protocollo e gamma variabili. Il sistema risponde con la catena completa:

  g0 (essenza/Fo, massa) -> gx (Fx operativa, qualita') ->
  gy (Fy novita') -> gf (quiete + certificazione + memoria cache)

Ogni 25 domande una e' la ripetizione esatta di una precedente: la
`MemoriaGF` condivisa deve dare hit senza ricalcolare (recall ~0).

Output in `output/outuput_test/` (nome richiesto dall'utente):
  - stress_500.csv / .md (una riga per domanda)
  - fig_stress_pannello1.png (tassi + istogrammi qualita'/novita')
  - fig_stress_pannello2.png (scatter eq/ea + qualita' nel tempo + hit cumulati)

Uso: `python -m src stress500 --n 500 --N 32 --T 0.10 --seed 7`
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .frammento_2d import Param, essenza, simula
from .frammento_gf import MemoriaGF, certifica_frammento

OUT_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", "outuput_test")


# ---------------------------------------------------------------------------
def genera_domande(n: int = 500, seed: int = 7) -> list[dict]:
    """500 cue: pos/ampiezza bump + seed/protocollo/gamma; ogni 25 una replica."""
    rng = np.random.default_rng(int(seed))
    domande: list[dict] = []
    for i in range(int(n)):
        if i >= 25 and i % 25 == 0:
            base = dict(domande[i - 25])
            base["id"] = i
            base["ripete"] = i - 25
            domande.append(base)
            continue
        domande.append({
            "id": i,
            "seed_domanda": int(rng.integers(0, 10_000)),
            "protocollo": "stimolo" if bool(rng.integers(0, 2)) else "rilassamento",
            "gamma": float(rng.choice([0.0, 0.02, 0.15])),
            "bump_x": float(rng.uniform(0.2, 0.8)),
            "bump_y": float(rng.uniform(0.2, 0.8)),
            "bump_amp": float(rng.uniform(0.5, 2.0)),
            "ripete": None,
        })
    return domande


def _bump(p: Param, ess: np.ndarray, dom: dict) -> np.ndarray:
    x = (np.arange(p.N) + 0.5) / p.N * p.L
    X, Y = np.meshgrid(x, x, indexing="ij")
    g = np.exp(-((X - dom["bump_x"] * p.L) ** 2
                 + (Y - dom["bump_y"] * p.L) ** 2) / (2 * 0.05 ** 2))
    return np.asarray(ess) + dom["bump_amp"] * g


def interroga(dom: dict, N: int = 32, T: float = 0.10,
              mem: MemoriaGF | None = None) -> dict:
    """Una domanda -> risposta completa g0/gx/gy/gf (+ cache)."""
    p = Param(N=N, seed=dom["seed_domanda"], gamma=dom["gamma"])
    ess = essenza(p)
    snap = simula(p, T=T, protocollo=dom["protocollo"], salva_ogni=10,
                  stato_iniziale={"Fx": _bump(p, ess, dom)})
    Fo, Fx, Fy = snap["F0"][-1], snap["Fx"][-1], snap["Fy"][-1]
    c = certifica_frammento(Fo, Fx, Fy, ess, p.dx, diag=snap["diag"])
    q = c["quiete"]
    hit = False
    if mem is not None:
        k = mem.chiave(Fo, Fx, Fy, ess, dx=p.dx)
        hit, _ = mem.richiama(k)
        if c["certificato"] and not hit:
            mem.salva(k, {"qualita": c["qualita"], "domanda": dom["id"]})
    d = snap["diag"]
    return {
        "id": dom["id"], "ripete": dom["ripete"],
        "protocollo": dom["protocollo"], "gamma": dom["gamma"],
        "massa_g0_drift": float(abs(d["massa0"][-1] - d["massa0"][0])),
        "qualita": float(c["qualita"]),
        "novita_rel": float(c["novita_rel"]),
        "err_fx_fo": float(q["err_fx_fo"]),
        "err_fo_ess": float(q["err_fo_ess"]),
        "quiete": bool(q["attivo"]), "cert": bool(c["certificato"]),
        "hit": bool(hit),
    }


# ---------------------------------------------------------------------------
def esegui(n: int = 500, N: int = 32, T: float = 0.10, seed: int = 7,
           out_dir: str = OUT_DIR_DEFAULT) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    domande = genera_domande(n=n, seed=seed)
    mem = MemoriaGF()
    righe = [interroga(d, N=N, T=T, mem=mem) for d in domande]

    fp_csv = os.path.join(out_dir, "stress_500.csv")
    with open(fp_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)

    Q = np.array([r["qualita"] for r in righe])
    Nv = np.array([r["novita_rel"] for r in righe])
    Eq = np.array([r["err_fx_fo"] for r in righe])
    Ea = np.array([r["err_fo_ess"] for r in righe])
    n_q = sum(r["quiete"] for r in righe)
    n_c = sum(r["cert"] for r in righe)
    n_h = sum(r["hit"] for r in righe)
    n_rep = sum(1 for r in righe if r["ripete"] is not None)
    hit_rep = sum(1 for r in righe if r["ripete"] is not None and r["hit"])

    # --- pannello 1: tassi + istogrammi
    fig, ax = plt.subplots(2, 2, figsize=(12.0, 7.5))
    ax[0, 0].bar(["quiete SI", "cert SI", "cache hit"],
                 [100.0 * n_q / n, 100.0 * n_c / n, 100.0 * n_h / n],
                 color=["#1f4e79", "#2e7d32", "#c47b17"])
    ax[0, 0].set_ylim(0, 100)
    ax[0, 0].set_ylabel("percentuale su %d domande" % n)
    ax[0, 0].set_title("Stress 500 domande: tassi g0->gf")
    ax[0, 0].grid(alpha=0.3, axis="y")
    ax[0, 1].hist(Q, bins=25, color="#1f4e79", alpha=0.8)
    ax[0, 1].axvline(0.40, color="red", ls="--", label="soglia gf 0.40")
    ax[0, 1].set_xlabel("qualita' gx"); ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("Qualita' (med %.3f)" % float(Q.mean()))
    ax[1, 0].hist(Nv, bins=25, color="#8b1e3f", alpha=0.8)
    ax[1, 0].axvline(0.30, color="red", ls="--", label="budget gf 0.30")
    ax[1, 0].set_xlabel("novita_rel gy"); ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("Novita' relativa (med %.4f)" % float(Nv.mean()))
    ax[1, 1].axis("off")
    ax[1, 1].text(0.05, 0.9, "quiete: %d/%d (%.1f%%)" % (n_q, n, 100.0 * n_q / n), fontsize=11)
    ax[1, 1].text(0.05, 0.7, "certificati: %d/%d (%.1f%%)" % (n_c, n, 100.0 * n_c / n), fontsize=11)
    ax[1, 1].text(0.05, 0.5, "cache hit: %d/%d (repliche %d/%d)" % (n_h, n, hit_rep, n_rep), fontsize=11)
    ax[1, 1].text(0.05, 0.3, "massa g0 drift max: %.2e" % max(r["massa_g0_drift"] for r in righe), fontsize=11)
    ax[1, 1].text(0.05, 0.1, "N=%d T=%.2f seed=%d" % (N, T, seed), fontsize=10)
    fig.suptitle("Frammento del veloce — stress test %d domande (g0->gx->gy->gf)" % n, fontsize=13)
    fig.tight_layout()
    f1 = os.path.join(out_dir, "fig_stress_pannello1.png")
    fig.savefig(f1, dpi=110)
    plt.close(fig)

    # --- pannello 2: scatter eq/ea + qualita' nel tempo + hit cumulati
    fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))
    col = np.array(["green" if r["cert"] else "red" for r in righe])
    ax[0].scatter(Eq, Ea, c=col, s=10, alpha=0.6)
    ax[0].axvline(0.60, color="k", ls="--", lw=1)
    ax[0].axhline(0.90, color="k", ls="--", lw=1)
    ax[0].set_xlabel("err_fx_fo (eps=0.60)")
    ax[0].set_ylabel("err_fo_ess (delta=0.90)")
    ax[0].set_title("Quiete: verde=cert, rosso=no")
    ax[0].grid(alpha=0.3)
    ax[1].plot(Q, color="#1f4e79", lw=0.8)
    ax[1].axhline(0.40, color="red", ls="--", label="soglia")
    ax[1].set_xlabel("domanda #"); ax[1].set_ylabel("qualita'")
    ax[1].set_title("Qualita' per domanda")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[2].plot(np.cumsum([r["hit"] for r in righe]), color="#c47b17", lw=1.2)
    ax[2].set_xlabel("domanda #"); ax[2].set_ylabel("hit cumulati")
    ax[2].set_title("Memoria: recall a costo ~0")
    ax[2].grid(alpha=0.3)
    fig.suptitle("Stress %d domande — quiete, qualita', memoria" % n, fontsize=13)
    fig.tight_layout()
    f2 = os.path.join(out_dir, "fig_stress_pannello2.png")
    fig.savefig(f2, dpi=110)
    plt.close(fig)

    fp_md = os.path.join(out_dir, "stress_500.md")
    with open(fp_md, "w") as f:
        f.write("# Stress test %d domande (g0->gf)\n\n" % n)
        f.write("N=%d T=%.2f seed=%d\n\n" % (N, T, seed))
        f.write("- quiete SI: %d/%d (%.1f%%)\n" % (n_q, n, 100.0 * n_q / n))
        f.write("- certificati: %d/%d (%.1f%%)\n" % (n_c, n, 100.0 * n_c / n))
        f.write("- cache hit: %d/%d (repliche %d/%d)\n" % (n_h, n, hit_rep, n_rep))
        f.write("- qualita' media: %.4f (min %.4f, max %.4f)\n"
                % (float(Q.mean()), float(Q.min()), float(Q.max())))
        f.write("- massa g0 drift max: %.2e\n" % max(r["massa_g0_drift"] for r in righe))
        f.write("- figure: fig_stress_pannello1.png, fig_stress_pannello2.png\n")
    print("stress %d: quiete %d cert %d hit %d (repliche %d/%d) Qmed %.3f" %
          (n, n_q, n_c, n_h, hit_rep, n_rep, float(Q.mean())))
    print("cartella:", out_dir)
    return {"n": n, "quiete": n_q, "cert": n_c, "hit": n_h,
            "hit_repliche": (hit_rep, n_rep), "q_media": float(Q.mean()),
            "out_dir": out_dir, "fig1": f1, "fig2": f2}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Stress test 500 domande g0->gf")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--T", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=OUT_DIR_DEFAULT)
    a = ap.parse_args(argv)
    esegui(n=a.n, N=a.N, T=a.T, seed=a.seed, out_dir=a.out)


if __name__ == "__main__":
    main()
