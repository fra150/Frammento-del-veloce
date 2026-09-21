"""Fase 15 — prove referee: baseline CL forti + plasticita' + OOD.

Risponde ai 3 punti del 21/09/2026 senza toccare il core esistente:
- Punto 1: Replay + EWC-lite nello STESSO linguaggio di campo (stessa Q,
  stessa sequenza cue di `esegui_test_1000`). Non torch, non MLP esterno:
  confronto mele-con-mele, riproducibile con solo numpy.
- Punto 2: misura plasticita' (tasso certificazione, Q media, tempo/ricordo,
  memoria bytes vs n slot).
- Punto 3: cue OOD (mix di due ricordi, rumore forte, regione mai vista).

Tutto deterministico (stocastico=False) come il test dei 1000.
"""

from __future__ import annotations

import time

import numpy as np

from .frammento_2d import Param, essenza, simula
from .rete_frammento import (
    ReteFrammento,
    ReteIngenuaCondivisa,
    genera_cue,
    cue_parziale,
    ricostruisci_associativo,
    _bump_iniziale,
    _qualita,
    _q_regione,
)


# ---------------------------------------------------------------------------
# Punto 1 — baseline continual learning forti (stesso campo, stessa Q)
# ---------------------------------------------------------------------------
class ReteReplay(ReteIngenuaCondivisa):
    """Baseline rehearsal: campo condiviso P + buffer di K residui passati.

    Ogni `impara`:
      r_new = Fx - ess
      buffer <- r_new (FIFO, max buffer_max)
      target = media(r_new + replay_k campioni dal buffer)
      P = (1-eta)*P + eta*target
    Il richiamo e' identico all'ingenua (deriva P - P_snapshot).
    Costo per passo O(replay_k), memoria O(buffer_max * N^2).
    """

    def __init__(self, N=32, T=0.10, eta=0.10, soglia_qualita=0.40,
                 buffer_max=50, replay_k=5, seed=0):
        super().__init__(N=N, T=T, eta=eta, soglia_qualita=soglia_qualita)
        self.buffer_max = int(buffer_max)
        self.replay_k = int(replay_k)
        self._rng = np.random.default_rng(int(seed))
        self._buf: list[np.ndarray] = []

    def impara(self, cue: dict) -> dict:
        p = Param(N=self.N, seed=int(cue["seed_domanda"]),
                  gamma=float(cue["gamma"]))
        ess = essenza(p)
        snap = simula(p, T=self.T, protocollo=cue["protocollo"],
                      salva_ogni=10, stocastico=False,
                      stato_iniziale={"Fx": _bump_iniziale(p, ess, cue)})
        Fx = np.asarray(snap["Fx"][-1])
        q = _qualita(Fx, ess)
        snap_P = self.P.copy()
        r_new = Fx - ess
        self._buf.append(r_new.copy())
        if len(self._buf) > self.buffer_max:
            self._buf.pop(0)
        k = min(self.replay_k, len(self._buf) - 1)
        if k > 0:
            idx = self._rng.choice(len(self._buf) - 1, size=k, replace=False)
            # il buffer include r_new in coda: campiona solo dai precedenti
            camp = [self._buf[i] for i in idx if i < len(self._buf) - 1]
            if camp:
                target = (r_new + np.mean(camp, axis=0)) / 2.0
            else:
                target = r_new
        else:
            target = r_new
        self.P = (1.0 - self.eta) * self.P + self.eta * target
        self.cue_note[int(cue["id"])] = dict(cue)
        self.record[int(cue["id"])] = {"Fx": Fx.copy(), "ess": ess.copy(),
                                       "q_cert": float(q),
                                       "P_snapshot": snap_P}
        return {"id": int(cue["id"]), "q": float(q)}


class ReteEWC(ReteIngenuaCondivisa):
    """Baseline EWC-lite in linguaggio di campo (regolarizzazione quadratica).

    Idea EWC (Kirkpatrick et al. 2017) tradotta in forma chiusa sul campo:
      loss(P) = ||P - r_new||^2 + lam * sum F_acc*(P - P_anchor)^2
      -> P_new = (r_new + lam*F_acc*P_anchor) / (1 + lam*F_acc)  (per-pixel)

    - F_new = importanza per-pixel ~ r_new^2 normalizzato a media 1.
    - F_acc si accumula (online EWC): F_acc += F_new dopo ogni task.
    - P_anchor = P dopo ogni task (ancora mobile, versione online).
    Con lam=0 si ricade nell'ingenua (a meno di eta=1 nel closed-form);
    con lam grande P si ancora al passato (meno forgetting, meno plasticita').
    Parametro eta qui = passo verso la soluzione chiusa:
      P = (1-eta)*P + eta*P_chiusa.

    ONESTO: e' un analogo concettuale, non l'EWC su gradienti di una rete
    torch. Serve a mostrare il trade-off forgetting/plasticita' a parita'
    di metrica Q e sequenza cue.
    """

    def __init__(self, N=32, T=0.10, eta=1.0, lam=10.0,
                 soglia_qualita=0.40, gamma=0.9):
        super().__init__(N=N, T=T, eta=eta, soglia_qualita=soglia_qualita)
        self.lam = float(lam)
        self.gamma = float(gamma)  # decadimento online (senza: rigidita' esplode)
        self.F_acc = np.zeros((self.N, self.N))
        self.P_anchor = np.zeros((self.N, self.N))

    def impara(self, cue: dict) -> dict:
        p = Param(N=self.N, seed=int(cue["seed_domanda"]),
                  gamma=float(cue["gamma"]))
        ess = essenza(p)
        snap = simula(p, T=self.T, protocollo=cue["protocollo"],
                      salva_ogni=10, stocastico=False,
                      stato_iniziale={"Fx": _bump_iniziale(p, ess, cue)})
        Fx = np.asarray(snap["Fx"][-1])
        q = _qualita(Fx, ess)
        snap_P = self.P.copy()
        r_new = Fx - ess
        if self.lam > 0 and np.any(self.F_acc > 0):
            denom = 1.0 + self.lam * self.F_acc
            P_chiusa = (r_new + self.lam * self.F_acc * self.P_anchor) / denom
        else:
            P_chiusa = r_new
        self.P = (1.0 - self.eta) * self.P + self.eta * P_chiusa
        # accumulo importanza online DECADUTO + sposta ancora
        # (senza gamma, F_acc cresce ~n e la plasticita' crolla: P si congela)
        f_new = r_new ** 2
        f_new = f_new / (f_new.mean() + 1e-12)
        self.F_acc = float(self.gamma) * self.F_acc + f_new
        self.P_anchor = self.P.copy()
        self.cue_note[int(cue["id"])] = dict(cue)
        self.record[int(cue["id"])] = {"Fx": Fx.copy(), "ess": ess.copy(),
                                       "q_cert": float(q),
                                       "P_snapshot": snap_P}
        return {"id": int(cue["id"]), "q": float(q)}


def esegui_confronto_cl(n_cert=40, n_nuove=60, N=16, T=0.05, seed=7,
                        replay_k=5, buffer_max=50, ewc_lam=1.0, ewc_eta=0.1,
                        shift=False) -> dict:
    """Stessa sequenza cue per 4 modelli: protetta / ingenua / replay / ewc.

    Ritorna per modello: max_degrado, distrutti, q_dopo media.
    La protetta certifica un sottoinsieme (gate gf); per gli altri si
    valutano gli stessi id certificati (fair: stessi ricordi, stessa Q).

    - `shift=False` (default storico): cue i.i.d. ovunque, amp 0.5-2.0.
      Regime a bassa interferenza (campi ~97% simili): tutti degradano poco.
    - `shift=True`: set A (n_cert, sinistra, amp 3-8) poi set B (n_nuove,
      destra, amp 3-8). Stesso disegno dello shift di classe di `assoc`:
      separa i metodi perche' il campo condiviso deve seguire B.
    """
    if shift:
        prime = genera_cue(int(n_cert), seed=seed, amp_range=(3.0, 8.0),
                           regione="sinistra")
        nuove = genera_cue(int(n_nuove), seed=seed + 1000,
                           amp_range=(3.0, 8.0), regione="destra")
        for k, c in enumerate(nuove):
            c["id"] = int(n_cert) + k
    else:
        tot = int(n_cert) + int(n_nuove)
        cue = genera_cue(tot, seed=seed)
        prime, nuove = cue[:int(n_cert)], cue[int(n_cert):]

    prot = ReteFrammento(N=N, T=T)
    ing = ReteIngenuaCondivisa(N=N, T=T)
    rep = ReteReplay(N=N, T=T, replay_k=replay_k, buffer_max=buffer_max,
                     seed=seed)
    ewc = ReteEWC(N=N, T=T, lam=ewc_lam, eta=ewc_eta)
    modelli = {"protetta": prot, "ingenua": ing, "replay": rep, "ewc": ewc}

    id_cert: list[int] = []
    id_nuove_cert: list[int] = []
    for c in prime:
        r = prot.impara(c)
        ing.impara(c)
        rep.impara(c)
        ewc.impara(c)
        if r["cert"]:
            id_cert.append(r["id"])
    for c in nuove:
        r = prot.impara(c)
        ing.impara(c)
        rep.impara(c)
        ewc.impara(c)
        if r["cert"]:
            id_nuove_cert.append(r["id"])

    out: dict = {"n_certificati": len(id_cert), "id_cert": id_cert,
                 "N": N, "T": T, "seed": seed, "shift": bool(shift),
                 "replay_k": int(replay_k), "buffer_max": int(buffer_max),
                 "ewc_lam": float(ewc_lam), "ewc_eta": float(ewc_eta),
                 "shift_rel": float(np.linalg.norm(rep.P - ing.P)
                                    / (np.linalg.norm(ing.P) + 1e-12))
                 if np.linalg.norm(ing.P) > 0 else 0.0}
    # protetta: richiamo esatto (ricalcola); altri: deriva da P
    degr_p, q_p, d_p = [], [], 0
    for i in id_cert:
        rr = prot.richiama_ricalcola(i)
        degr_p.append(rr["degrado"])
        q_p.append(rr["q_recall"])
        d_p += int(rr["distrutto"])
    out["protetta"] = {"max_degrado": float(np.max(degr_p)) if degr_p else 0.0,
                       "distrutti": int(d_p),
                       "q_media": float(np.mean(q_p)) if q_p else 0.0}
    # plasticita' su B: Q media di richiamo sulle nuove certificate (stesse id)
    qb_p = []
    for i in id_nuove_cert:
        try:
            rr = prot.richiama_ricalcola(i)
            qb_p.append(rr["q_recall"])
        except AssertionError:
            pass
    out["protetta"]["q_nuove"] = float(np.mean(qb_p)) if qb_p else 0.0
    for nome in ("ingenua", "replay", "ewc"):
        m = modelli[nome]
        degr, q2, d = [], [], 0
        for i in id_cert:
            ri = m.richiama(i)
            degr.append(ri["degrado"])
            q2.append(ri["q_recall"])
            d += int(ri["distrutto"])
        qb = []
        for i in id_nuove_cert:
            if i in m.record:
                try:
                    qb.append(m.richiama(i)["q_recall"])
                except KeyError:
                    pass
        out[nome] = {"max_degrado": float(np.max(degr)) if degr else 0.0,
                     "distrutti": int(d),
                     "q_media": float(np.mean(q2)) if q2 else 0.0,
                     "degrado_medio": float(np.mean(degr)) if degr else 0.0,
                     "q_nuove": float(np.mean(qb)) if qb else 0.0}
    # costi onesti: memoria extra oltre il modello base (bytes)
    n2 = N * N
    out["costi_bytes"] = {
        "protetta_slot": len(prot.slot) * 4 * n2 * 8,
        "replay_buf": len(rep._buf) * n2 * 8,
        "ewc_extra": 2 * n2 * 8,  # F_acc + P_anchor
    }
    out["memoria_replay_buffer"] = len(rep._buf)
    return out


def sweep_pareto_cl(n_cert=40, n_nuove=60, N=16, T=0.05, seed=7,
                    shift=True) -> list[dict]:
    """Sweep lam/buffer a parita' di sequenza (shift di classe).

    Configurazioni:
      ingenua | replay K=5/10/20 (buf 50) | ewc lam=0.5/1/2/5 (eta 0.5)
    Ritorna righe con forgetting (max/medio, distrutti) + costi (bytes).
    La protetta e' invariante (riportata in ogni riga per riferimento).
    """
    configs: list[dict] = [{"nome": "ingenua", "replay_k": 0,
                            "buffer_max": 0, "ewc_lam": 0.0}]
    # replay: K fisso 10, buffer che trattiene tutto vs FIFO corto che
    # dimentica A dopo lo shift (punto Pareto: per reggere lo shift il
    # buffer deve crescere come gli slot della protetta)
    for buf, k in ((30, 10), (200, 10), (200, 20)):
        configs.append({"nome": f"replay_K{k}_B{buf}", "replay_k": k,
                        "buffer_max": buf, "ewc_lam": 0.0})
    for lam in (0.1, 0.2, 0.5, 1.0):
        configs.append({"nome": f"ewc_l{lam:g}", "replay_k": 0,
                        "buffer_max": 0, "ewc_lam": float(lam)})
    righe: list[dict] = []
    for cf in configs:
        if cf["nome"].startswith("replay"):
            r = esegui_confronto_cl(n_cert=n_cert, n_nuove=n_nuove, N=N,
                                    T=T, seed=seed, shift=shift,
                                    replay_k=cf["replay_k"],
                                    buffer_max=cf["buffer_max"])
            m = r["replay"]
            costo = r["costi_bytes"]["replay_buf"]
        elif cf["nome"].startswith("ewc"):
            r = esegui_confronto_cl(n_cert=n_cert, n_nuove=n_nuove, N=N,
                                    T=T, seed=seed, shift=shift,
                                    ewc_lam=cf["ewc_lam"])
            m = r["ewc"]
            costo = r["costi_bytes"]["ewc_extra"]
        else:
            r = esegui_confronto_cl(n_cert=n_cert, n_nuove=n_nuove, N=N,
                                    T=T, seed=seed, shift=shift)
            m = r["ingenua"]
            costo = 0
        righe.append({
            "modello": cf["nome"],
            "n_cert": r["n_certificati"],
            "max_degrado": m["max_degrado"],
            "degrado_medio": m.get("degrado_medio", 0.0),
            "distrutti": m["distrutti"],
            "q_media": m["q_media"],
            "q_nuove": m.get("q_nuove", 0.0),
            "costo_bytes": int(costo),
            "prot_max_degr": r["protetta"]["max_degrado"],
            "prot_distrutti": r["protetta"]["distrutti"],
        })
    return righe


def salva_report_cl(righe: list[dict], out_dir: str | None = None,
                    N=16, T=0.05, seed=7, shift=True) -> dict:
    """CSV + md + fig15 (Pareto forgetting vs costo). Ritorna path."""
    import csv as _csv
    import os as _os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "output", "output_test")
    _os.makedirs(out_dir, exist_ok=True)
    fp_csv = _os.path.join(out_dir, "fase15_cl_pareto.csv")
    with open(fp_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["modello", "n_cert", "max_degrado", "degrado_medio",
                    "distrutti", "q_media", "q_nuove", "costo_bytes"])
        for r in righe:
            w.writerow([r["modello"], r["n_cert"],
                        f"{r['max_degrado']:.6f}",
                        f"{r['degrado_medio']:.6f}", r["distrutti"],
                        f"{r['q_media']:.6f}", f"{r.get('q_nuove', 0.0):.6f}",
                        r["costo_bytes"]])
    fp_md = _os.path.join(out_dir, "fase15_cl_pareto.md")
    with open(fp_md, "w") as f:
        f.write("# Fase 15 — Pareto CL con shift di classe (stessa Q)\n\n")
        f.write(f"N={N} T={T} seed={seed} shift={shift}\n\n")
        f.write("| modello | max degr | medio | distrutti | Q vecchie | Q nuove | costo |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in righe:
            f.write(f"| {r['modello']} | {r['max_degrado']:.4f} | "
                    f"{r['degrado_medio']:.4f} | {r['distrutti']}/{r['n_cert']} "
                    f"| {r['q_media']:.4f} | {r.get('q_nuove', 0.0):.4f} "
                    f"| {r['costo_bytes']/1024:.1f} KB |\n")
        f.write("\nProtetta: max degr 0, distrutti 0 (strutturale). "
                "Costo protetta = 4 campi/slot (vedi plasticita').\n")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    xs = [r["costo_bytes"] / 1024.0 for r in righe]
    ys = [r["max_degrado"] for r in righe]
    for x, y, r in zip(xs, ys, righe):
        ax.scatter([x], [y], s=90)
        ax.text(x, y, f" {r['modello']}", fontsize=9, va="bottom")
    ax.axhline(0.0, color="green", ls="--", lw=1, label="protetta (0)")
    ax.axhline(0.02, color="red", ls=":", lw=1, label="soglia distruzione 0.02")
    ax.set_xlabel("costo memoria extra (KB)")
    ax.set_ylabel("max degrado su ricordi A dopo shift B")
    ax.set_title(f"Pareto CL con shift (N={N}, seed={seed})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ffig = _os.path.join(out_dir, "fig15_cl_pareto.png")
    fig.savefig(ffig, dpi=120)
    plt.close(fig)
    print(f"cl-pareto: {len(righe)} config | fig {ffig}")
    return {"csv": fp_csv, "md": fp_md, "fig": ffig}


# ---------------------------------------------------------------------------
# Punto 2 — plasticita': capacita' vs Q, memoria vs slot, costo/ricordo
# ---------------------------------------------------------------------------
def misura_plasticita(n_list=(25, 50, 100, 200), N=16, T=0.05,
                      seed=7, amp_range=(0.5, 2.0)) -> list[dict]:
    """Per ogni n: impara n cue, misura tasso cert, Q media, tempo, memoria."""
    righe: list[dict] = []
    for n in n_list:
        cue = genera_cue(int(n), seed=seed, amp_range=tuple(amp_range))
        r = ReteFrammento(N=N, T=T)
        t0 = time.perf_counter()
        n_cert = 0
        qs: list[float] = []
        for c in cue:
            o = r.impara(c)
            if o["cert"]:
                n_cert += 1
                qs.append(o["q"])
        dt = time.perf_counter() - t0
        # memoria: 4 campi float64 per slot (Fx,Fo,Fy,ess)
        per_slot = 4 * N * N * 8
        mem_bytes = len(r.slot) * per_slot
        righe.append({
            "n_richieste": int(n),
            "n_certificati": int(n_cert),
            "tasso_cert": float(n_cert / max(1, int(n))),
            "q_media": float(np.mean(qs)) if qs else 0.0,
            "tempo_tot_s": float(dt),
            "tempo_per_ricordo_ms": float(dt / max(1, int(n)) * 1000.0),
            "memoria_bytes": int(mem_bytes),
            "memoria_kb": float(mem_bytes / 1024.0),
            "capacita_slot": int(len(r.slot)),
        })
    return righe


def salva_report_plasticita(righe: list[dict], out_dir: str | None = None,
                            N=16, T=0.05, seed=7) -> dict:
    """CSV + md + fig16 (tasso cert, Q, tempo/ricordo, memoria vs n)."""
    import csv as _csv
    import os as _os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "output", "output_test")
    _os.makedirs(out_dir, exist_ok=True)
    fp_csv = _os.path.join(out_dir, "fase15_plasticita.csv")
    with open(fp_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["n_richieste", "n_certificati", "tasso_cert", "q_media",
                    "tempo_tot_s", "tempo_per_ricordo_ms", "memoria_kb"])
        for r in righe:
            w.writerow([r["n_richieste"], r["n_certificati"],
                        f"{r['tasso_cert']:.4f}", f"{r['q_media']:.6f}",
                        f"{r['tempo_tot_s']:.3f}",
                        f"{r['tempo_per_ricordo_ms']:.3f}",
                        f"{r['memoria_kb']:.1f}"])
    fp_md = _os.path.join(out_dir, "fase15_plasticita.md")
    with open(fp_md, "w") as f:
        f.write("# Fase 15 — Plasticita' (capacita' vs Q, memoria vs slot)\n\n")
        f.write(f"N={N} T={T} seed={seed}\n\n")
        f.write("| n richieste | certificati | tasso | Q media | ms/ricordo | memoria |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in righe:
            f.write(f"| {r['n_richieste']} | {r['n_certificati']} | "
                    f"{r['tasso_cert']:.3f} | {r['q_media']:.4f} | "
                    f"{r['tempo_per_ricordo_ms']:.2f} | "
                    f"{r['memoria_kb']:.0f} KB |\n")
        f.write("\nFattura dello zero-forgetting strutturale: memoria lineare "
                "4 campi float64/slot, tempo lineare per ricordo.\n")
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 7.5))
    xs = [r["n_richieste"] for r in righe]
    ax[0, 0].plot(xs, [r["tasso_cert"] for r in righe], "o-", color="#1f4e79")
    ax[0, 0].set_xlabel("n richieste")
    ax[0, 0].set_ylabel("tasso certificazione")
    ax[0, 0].set_title("A) Quanti ricordi passano il gate gf")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 1].plot(xs, [r["q_media"] for r in righe], "o-", color="#2e7d32")
    ax[0, 1].set_xlabel("n richieste")
    ax[0, 1].set_ylabel("Q media certificati")
    ax[0, 1].set_title("B) Qualita' vs capacita' (piatta = non satura)")
    ax[0, 1].grid(alpha=0.3)
    ax[1, 0].plot(xs, [r["tempo_per_ricordo_ms"] for r in righe],
                  "o-", color="#c47b17")
    ax[1, 0].set_xlabel("n richieste")
    ax[1, 0].set_ylabel("ms / ricordo")
    ax[1, 0].set_title("C) Costo computazionale per ricordo")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 1].plot(xs, [r["memoria_kb"] for r in righe], "o-",
                  color="#8b1e3f", label="misurata")
    # retta teorica 4*N*N*8 byte/slot
    teo = [x * 4 * N * N * 8 / 1024.0 for x in xs]
    ax[1, 1].plot(xs, teo, "--", color="gray", lw=1, label="teorica lineare")
    ax[1, 1].set_xlabel("n richieste")
    ax[1, 1].set_ylabel("KB")
    ax[1, 1].set_title("D) Memoria vs slot (lineare)")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3)
    fig.suptitle(f"Plasticita' protetta (N={N}, T={T}, seed={seed})",
                 fontsize=13)
    fig.tight_layout()
    ffig = _os.path.join(out_dir, "fig16_plasticita.png")
    fig.savefig(ffig, dpi=120)
    plt.close(fig)
    print(f"plasticita: {len(righe)} punti | fig {ffig}")
    return {"csv": fp_csv, "md": fp_md, "fig": ffig}


# ---------------------------------------------------------------------------
# Punto 3 — OOD: mix di ricordi, rumore forte, regione mai vista
# + retrieval (selezione del prior, il passo rischioso)
# ---------------------------------------------------------------------------
def seleziona_prior(rete: ReteFrammento, cue: np.ndarray,
                    vis: np.ndarray) -> dict:
    """Retrieval: quale slot spiega meglio la parte VISIBILE del cue?

    Score per slot = MSE(cue[vis], Fx_slot[vis]). Ritorna best_id,
    best_mse, second_mse e margine = (second-best)/best (confidenza:
    margine ~0 = ambiguo, >>0 = decisione netta).
    Non usa mai la regione mancante: niente oracolo.
    """
    cue_a = np.asarray(cue, dtype=float)
    vis_a = np.asarray(vis, dtype=bool)
    scored: list[tuple[float, int]] = []
    for cid, rec in rete.slot.items():
        d = cue_a[vis_a] - np.asarray(rec["Fx"], dtype=float)[vis_a]
        scored.append((float(np.mean(d ** 2)), int(cid)))
    scored.sort()
    best_mse, best_id = scored[0]
    second_mse = scored[1][0] if len(scored) > 1 else float("nan")
    margine = ((second_mse - best_mse) / (best_mse + 1e-12)
               if len(scored) > 1 else 0.0)
    return {"best_id": int(best_id), "best_mse": float(best_mse),
            "second_mse": float(second_mse), "margine": float(margine),
            "n_candidati": len(scored)}


def test_ood_mix_retrieval(N=16, T=0.05, seed=7,
                           alpha_list=(0.0, 0.25, 0.5, 0.75, 1.0),
                           frazione=0.5, rumore=0.0) -> list[dict]:
    """Mix A+B mai visto + retrieval del prior sulla parte visibile.

    Train: 4 ricordi sx (A) + 4 dx (B), bump 3-8. Poi cue = mix convessa
    di un ricordo A e uno B, resa parziale (blocco) + rumore opzionale.
    Il retrieval sceglie lo slot sulla parte visibile; la ricostruzione
    usa il Fo dello slot recuperato (winner-take-all reale, non oracolo).
    Metriche: chi vince (A/B), margine di confidenza, q vs mix e vs vero.
    Se i campi sono ~97% simili, il margine sara' ~0 e a alpha=0.5 il
    retrieval sara' quasi chance: da riportare onesto.
    """
    cueA = genera_cue(4, seed=seed, amp_range=(3.0, 8.0), regione="sinistra")
    cueB = genera_cue(4, seed=seed + 1000, amp_range=(3.0, 8.0),
                      regione="destra")
    for k, c in enumerate(cueB):
        c["id"] = 4 + k  # id distinti: senza offset B sovrascriverebbe A
    rete = ReteFrammento(N=N, T=T)
    for c in cueA + cueB:
        rete.impara(c)
    ids = sorted(rete.slot.keys())
    a_ids = [i for i in ids if i < 4]
    b_ids = [i for i in ids if i >= 4]
    if not a_ids or not b_ids:
        return []
    Fa = np.asarray(rete.slot[a_ids[0]]["Fx"])
    Fb = np.asarray(rete.slot[b_ids[0]]["Fx"])
    dx = 1.0 / N
    rng = np.random.default_rng(seed + 77)
    righe: list[dict] = []
    for al in alpha_list:
        mix = miscela_ricordi(Fa, Fb, al)
        cue, vis = cue_parziale(mix, frazione=frazione, tipo="blocco",
                                rumore=float(rumore),
                                seed=int(rng.integers(0, 10_000)))
        sel = seleziona_prior(rete, cue, vis)
        rec = rete.slot[sel["best_id"]]
        Fr = ricostruisci_associativo(cue, vis, rec["Fo"], dx, modo="residuo")
        righe.append({
            "alpha": float(al),
            "vincitore": ("A" if sel["best_id"] in a_ids else
                          "B" if sel["best_id"] in b_ids else "?"),
            "best_id": sel["best_id"],
            "margine": sel["margine"],
            "q_vs_mix": float(_qualita(Fr, mix)),
            "q_mask_vs_mix": float(_q_regione(Fr, mix, ~vis)),
            "q_vs_A": float(_qualita(Fr, Fa)),
            "q_vs_B": float(_qualita(Fr, Fb)),
        })
    return righe


def test_ood_rumore_retrieval(N=16, T=0.05, seed=7,
                              rumori=(0.0, 0.5, 1.0, 2.0),
                              frazione=0.5) -> list[dict]:
    """Retrieval del ricordo giusto al crescere del rumore (stesso set)."""
    cue = genera_cue(8, seed=seed, amp_range=(3.0, 8.0))
    rete = ReteFrammento(N=N, T=T)
    for c in cue:
        rete.impara(c)
    ids = sorted(rete.slot.keys())
    righe: list[dict] = []
    for rum in rumori:
        ok, marg = 0, []
        for cid in ids:
            rec = rete.slot[cid]
            cp, vis = cue_parziale(rec["Fx"], frazione=frazione,
                                   tipo="blocco", rumore=float(rum),
                                   seed=cid * 13 + 1)
            sel = seleziona_prior(rete, cp, vis)
            ok += int(sel["best_id"] == cid)
            marg.append(sel["margine"])
        righe.append({"rumore": float(rum),
                      "accuratezza": float(ok / max(1, len(ids))),
                      "margine_medio": float(np.mean(marg)) if marg else 0.0})
    return righe


def salva_report_ood(mix_righe: list[dict], rum_righe: list[dict],
                     out_dir: str | None = None,
                     N=16, T=0.05, seed=7) -> dict:
    """CSV + md + fig17 (retrieval OOD: mix + rumore)."""
    import csv as _csv
    import os as _os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "output", "output_test")
    _os.makedirs(out_dir, exist_ok=True)
    fp_csv = _os.path.join(out_dir, "fase15_ood.csv")
    with open(fp_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["tipo", "alpha_rumore", "vincitore_acc", "margine_q",
                    "q1", "q2"])
        for r in mix_righe:
            w.writerow(["mix", f"{r['alpha']:.2f}", r["vincitore"],
                        f"{r['margine']:.6f}", f"{r['q_vs_mix']:.6f}",
                        f"{r['q_mask_vs_mix']:.6f}"])
        for r in rum_righe:
            w.writerow(["rumore", f"{r['rumore']:.2f}",
                        f"{r['accuratezza']:.3f}",
                        f"{r['margine_medio']:.6f}", "", ""])
    fp_md = _os.path.join(out_dir, "fase15_ood.md")
    with open(fp_md, "w") as f:
        f.write("# Fase 15 — OOD con retrieval (prior non oracolo)\n\n")
        f.write(f"N={N} T={T} seed={seed}\n\n")
        f.write("## Mix A+B (alpha=1 -> puro A, 0 -> puro B)\n\n")
        f.write("| alpha | vincitore | margine | q vs mix | q vs A | q vs B |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in mix_righe:
            f.write(f"| {r['alpha']:.2f} | {r['vincitore']} | "
                    f"{r['margine']:.2e} | {r['q_vs_mix']:.4f} | "
                    f"{r['q_vs_A']:.4f} | {r['q_vs_B']:.4f} |\n")
        f.write("\n## Rumore (retrieval dello stesso ricordo)\n\n")
        f.write("| rumore | accuratezza | margine medio |\n")
        f.write("|---|---|---|\n")
        for r in rum_righe:
            f.write(f"| {r['rumore']:.2f} | {r['accuratezza']:.3f} | "
                    f"{r['margine_medio']:.2e} |\n")
        f.write("\nLettura: margine ~1e-6 = decisione ambigua (campi ~97% "
                "simili, il visibile non discrimina). Il completamento "
                "ricostruisce, non ragiona: il prior recuperato vince.\n")
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.5))
    al = [r["alpha"] for r in mix_righe]
    ax[0].plot(al, [r["q_vs_A"] for r in mix_righe], "o-", label="q vs A")
    ax[0].plot(al, [r["q_vs_B"] for r in mix_righe], "s-", label="q vs B")
    ax[0].plot(al, [r["q_vs_mix"] for r in mix_righe], "^-",
               label="q vs mix")
    for x, r in zip(al, mix_righe):
        ax[0].text(x, r["q_vs_mix"], f" {r['vincitore']}", fontsize=9)
    ax[0].set_xlabel("alpha (1=puro A)")
    ax[0].set_ylabel("qualita'")
    ax[0].set_title("A) Mix OOD: chi vince dopo retrieval")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot([r["rumore"] for r in rum_righe],
               [r["accuratezza"] for r in rum_righe], "o-", color="#8b1e3f")
    ax[1].set_xlabel("rumore (x std)")
    ax[1].set_ylabel("accuratezza retrieval")
    ax[1].set_title("B) Retrieval vs rumore")
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"OOD con retrieval (N={N}, seed={seed})", fontsize=13)
    fig.tight_layout()
    ffig = _os.path.join(out_dir, "fig17_ood.png")
    fig.savefig(ffig, dpi=120)
    plt.close(fig)
    print(f"ood: mix {len(mix_righe)} + rumore {len(rum_righe)} | fig {ffig}")
    return {"csv": fp_csv, "md": fp_md, "fig": ffig}
def miscela_ricordi(Fx_a: np.ndarray, Fx_b: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """Cue composizionale mai vista: combinazione convessa di due stati."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * np.asarray(Fx_a, dtype=float) + (1.0 - a) * np.asarray(Fx_b, dtype=float)


def test_ood_mix(N=16, T=0.05, seed=7, alpha_list=(0.0, 0.25, 0.5, 0.75, 1.0),
                 frazione=0.5) -> list[dict]:
    """Due ricordi lontani (sx vs dx, bump grandi), poi cue = loro miscela.

    Ricostruzione residuo con prior = core del ricordo A.
    Metriche: q vs A, q vs B, q vs miscela ideale. Se la rete 'ragiona'
    dovrebbe vincere il piu' vicino (winner-take-all); se interpola,
    q_mix resta alta e q_A/q_B si dimezzano a alpha=0.5.
    """
    cueA = genera_cue(4, seed=seed, amp_range=(3.0, 8.0), regione="sinistra")
    cueB = genera_cue(4, seed=seed + 1000, amp_range=(3.0, 8.0),
                      regione="destra")
    for k, c in enumerate(cueB):
        c["id"] = 100 + k  # id distinti: senza offset B sovrascriverebbe A
    rete = ReteFrammento(N=N, T=T)
    for c in cueA + cueB:
        rete.impara(c)
    ids = sorted(rete.slot.keys())
    if len(ids) < 2:
        return []
    a_id, b_id = ids[0], ids[-1]
    Fa = np.asarray(rete.slot[a_id]["Fx"])
    Fb = np.asarray(rete.slot[b_id]["Fx"])
    FoA = np.asarray(rete.slot[a_id]["Fo"])
    dx = 1.0 / N
    righe: list[dict] = []
    rng = np.random.default_rng(seed)
    for al in alpha_list:
        mix = miscela_ricordi(Fa, Fb, al)
        # cue parziale della miscela (blocco centrale mancante)
        cue, vis = cue_parziale(mix, frazione=frazione, tipo="blocco",
                                rumore=0.0, seed=int(rng.integers(0, 10_000)))
        rec = ricostruisci_associativo(cue, vis, FoA, dx, modo="residuo")
        righe.append({
            "alpha": float(al),
            "q_vs_A": float(_qualita(rec, Fa)),
            "q_vs_B": float(_qualita(rec, Fb)),
            "q_vs_mix": float(_qualita(rec, mix)),
            "q_mask_vs_mix": float(_q_regione(rec, mix, ~vis)),
        })
    return righe


def test_ood_rumore(N=16, T=0.05, seed=7,
                    rumori=(0.0, 0.25, 0.5, 1.0, 2.0),
                    frazione=0.5) -> list[dict]:
    """Sweep rumore sulla cue parziale: dove crolla il residuo?"""
    cue = genera_cue(6, seed=seed, amp_range=(3.0, 8.0))
    rete = ReteFrammento(N=N, T=T)
    for c in cue:
        rete.impara(c)
    ids = sorted(rete.slot.keys())[:3]
    dx = 1.0 / N
    righe: list[dict] = []
    for rum in rumori:
        qs = []
        for cid in ids:
            rec = rete.slot[cid]
            cp, vis = cue_parziale(rec["Fx"], frazione=frazione, tipo="blocco",
                                   rumore=float(rum), seed=cid * 13 + 1)
            Fr = ricostruisci_associativo(cp, vis, rec["Fo"], dx,
                                          modo="residuo")
            qs.append(_q_regione(Fr, rec["Fx"], ~vis))
        righe.append({"rumore": float(rum),
                      "q_mask_media": float(np.mean(qs)) if qs else 0.0,
                      "q_mask_min": float(np.min(qs)) if qs else 0.0})
    return righe


def test_ood_regione(N=16, T=0.05, seed=7) -> dict:
    """Bump in regione mai vista (centro-alto) dopo training solo sx/dx."""
    cueA = genera_cue(8, seed=seed, amp_range=(3.0, 8.0), regione="sinistra")
    rete = ReteFrammento(N=N, T=T)
    for c in cueA:
        rete.impara(c)
    ids = sorted(rete.slot.keys())
    # nuova cue: bump al centro-alto, amp fuori range (10.0)
    p = Param(N=N, seed=999)
    ess = essenza(p)
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    g = np.exp(-((X - 0.5) ** 2 + (Y - 0.9) ** 2) / (2 * 0.05 ** 2))
    Fx_nuovo = np.asarray(ess) + 10.0 * g
    dx = 1.0 / N
    cue, vis = cue_parziale(Fx_nuovo, frazione=0.5, tipo="blocco", seed=3)
    # prior = core del ricordo piu' vicino (qui il primo)
    Fo = np.asarray(rete.slot[ids[0]]["Fo"]) if ids else np.asarray(ess)
    Fr = ricostruisci_associativo(cue, vis, Fo, dx, modo="residuo")
    return {"q_vs_nuovo": float(_qualita(Fr, Fx_nuovo)),
            "q_mask_vs_nuovo": float(_q_regione(Fr, Fx_nuovo, ~vis)),
            "n_train": len(ids)}
