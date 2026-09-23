"""Fase 16 — sonno + dimenticare apposta + metriche CL + retrieval repair.

Risponde alle 4 direzioni del 21-23/09/2026 senza toccare il core esistente.
Solo numpy + matplotlib (figure) + hashlib/std. Tutto deterministico
(stocastico=False), come il test dei 1000.

1. SONNO (gist semantico offline):
   distilla dagli slot congelati un prior condiviso `gist = media(Fx)`
   SENZA scrivere negli slot (lettura sola). Politica d'uso onesta:
   se il retrieval winner-take-all e' ambiguo (margine < soglia), usa il
   gist; altrimenti tiene il vincitore. Verifica d'invarianza: hash slot
   + checksum nucleo prima/dopo identici.

2. DIMENTICARE APPOSTA (eviction esplicita):
   cancellazione (mai sovrascrittura) per eta' / Q / uso, con misura di
   copertura persa, Q media dei rimasti e byte risparmiati. Trasforma la
   memoria lineare in curva costo/perdita.

3. METRICHE CL STANDARD (BWT/FWT, Lopez-Paz adattato a 2 task A->B):
   BWT = Q(A|dopo B) - Q(A|dopo A); FWT = Q(B|A->B) - Q(B|B-solo).
   Protetta attesa BWT=FWT=0 (isolamento); condivise negative/positive.

4. RETRIEVAL REPAIR (validazione interna + coarse-to-fine):
   il residuo con ancoraggio rende banale la consistenza sul visibile
   (tutti i candidati hanno errore 0 sul visibile), quindi la gara tra
   prior si fa in validazione: si tiene da parte una frazione del visibile
   (val), si ricostruisce dal fit e si sceglie col MSE sul val (mai
   sull'occulto: niente oracolo). Coarse-to-fine: scrematura a griglia
   grossolana, top-k, poi gara di validazione a risoluzione piena.
"""

from __future__ import annotations

import csv as _csv
import os as _os

import numpy as np

from .fase15 import ReteEWC, ReteReplay
from .rete_frammento import (
    ReteFrammento,
    ReteIngenuaCondivisa,
    checksum_arr,
    cue_parziale,
    genera_cue,
    ricostruisci_associativo,
    _qualita,
    _q_regione,
)
from .fase15 import seleziona_prior

OUT_DIR_DEFAULT = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "output", "output_test")

PER_SLOT_CAMPI = 4  # Fx, Fo, Fy, ess (float64) come in fase15


# ===========================================================================
# 1. SONNO
# ===========================================================================
def hash_slot(rete: ReteFrammento) -> dict[int, str]:
    """Foto degli slot: checksum Fx per id (per provare il non-tocco)."""
    return {int(cid): checksum_arr(np.asarray(rec["Fx"]))
            for cid, rec in rete.slot.items()}


def distilla_gist(rete: ReteFrammento) -> dict:
    """Media degli Fx certificati = prototipo condiviso (lettura sola).

    Con bump casuali il gist e' volutamente sfocato: e' pooling, non
    ragionamento. Ritorna gist (prior), gist_Fo, residuo medio, n, costo.
    Non modifica rete (nessuna scrittura in slot/nucleo/memoria).
    """
    ids = sorted(rete.slot.keys())
    assert ids, "sonno senza ricordi: distilla_gist richiede slot non vuoti"
    N = rete.N
    stack_fx = np.stack([np.asarray(rete.slot[i]["Fx"], dtype=float)
                         for i in ids], axis=0)
    stack_fo = np.stack([np.asarray(rete.slot[i]["Fo"], dtype=float)
                         for i in ids], axis=0)
    gist = np.mean(stack_fx, axis=0)
    gist_fo = np.mean(stack_fo, axis=0)
    residuo_medio = np.mean(stack_fx - stack_fo, axis=0)
    # dispersione tra slot (quanto il gist e' sfocato)
    std_pixel = float(np.mean(np.std(stack_fx, axis=0)))
    return {
        "gist": gist,
        "gist_Fo": gist_fo,
        "residuo_medio": residuo_medio,
        "n_slot": len(ids),
        "N": int(N),
        "std_pixel": std_pixel,
        "norma_gist": float(np.linalg.norm(gist)),
        "costo_bytes": int(N * N * 8),
    }


def sonno(rete: ReteFrammento) -> dict:
    """Fase di sonno offline: verifica invarianza + distilla gist.

    Garanzie: checksum nucleo + hash di ogni slot identici prima/dopo.
    Non scrive in slot/nucleo/memoria (solo lettura + calcolo gist).
    """
    ck_prima = rete.verifica_nucleo()
    h_prima = hash_slot(rete)
    g = distilla_gist(rete)
    ck_dopo = rete.verifica_nucleo()
    h_dopo = hash_slot(rete)
    intatto = bool(ck_dopo["ok"] and ck_prima["checksum"] == ck_dopo["checksum"]
                   and h_prima == h_dopo)
    return {"gist": g["gist"], "dettagli": g, "intatto": intatto,
            "checksum": ck_dopo["checksum"], "n_slot": g["n_slot"]}


def valuta_sonno_mix(N: int = 16, T: float = 0.05, seed: int = 7,
                     soglia_margine: float = 0.05,
                     alpha_list=(0.0, 0.25, 0.5, 0.75, 1.0),
                     frazione: float = 0.5) -> list[dict]:
    """Mix A+B mai visto: winner vs gist con politica a soglia.

    Train 4 sx + 4 dx (bump 3-8). Per ogni alpha: cue = mix, resa parziale,
    retrieval sul visibile. Se margine < soglia -> usa gist, altrimenti
    tiene il vincitore. Metriche q vs mix per entrambi + scelta.
    """
    cueA = genera_cue(4, seed=seed, amp_range=(3.0, 8.0), regione="sinistra")
    cueB = genera_cue(4, seed=seed + 1000, amp_range=(3.0, 8.0),
                      regione="destra")
    for k, c in enumerate(cueB):
        c["id"] = 4 + k
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
    s = sonno(rete)
    gist = np.asarray(s["gist"])
    assert s["intatto"], "sonno ha toccato gli slot (non deve mai accadere)"
    dx = 1.0 / N
    rng = np.random.default_rng(seed + 913)
    righe: list[dict] = []
    for al in alpha_list:
        mix = float(al) * Fa + (1.0 - float(al)) * Fb
        cue, vis = cue_parziale(mix, frazione=frazione, tipo="blocco",
                                rumore=0.0,
                                seed=int(rng.integers(0, 10_000)))
        sel = seleziona_prior(rete, cue, vis)
        rec_w = rete.slot[sel["best_id"]]
        Fr_w = ricostruisci_associativo(cue, vis, rec_w["Fo"], dx,
                                        modo="residuo")
        Fr_g = ricostruisci_associativo(cue, vis, gist, dx, modo="residuo")
        uso_gist = bool(sel["margine"] < float(soglia_margine))
        righe.append({
            "alpha": float(al),
            "vincitore": ("A" if sel["best_id"] in a_ids else
                          "B" if sel["best_id"] in b_ids else "?"),
            "margine": float(sel["margine"]),
            "uso_gist": uso_gist,
            "q_winner_vs_mix": float(_qualita(Fr_w, mix)),
            "q_gist_vs_mix": float(_qualita(Fr_g, mix)),
            "q_scelta_vs_mix": float(_qualita(Fr_g if uso_gist else Fr_w,
                                             mix)),
        })
    return righe


def valuta_sonno_rumore(N: int = 16, T: float = 0.05, seed: int = 7,
                        soglia_margine: float = 0.05,
                        rumori=(0.0, 0.5, 1.0, 2.0),
                        frazione: float = 0.5) -> list[dict]:
    """Retrieval con rumore: qualita' winner vs gist (stesso set, 8 cue)."""
    cue = genera_cue(8, seed=seed, amp_range=(3.0, 8.0))
    rete = ReteFrammento(N=N, T=T)
    for c in cue:
        rete.impara(c)
    ids = sorted(rete.slot.keys())
    s = sonno(rete)
    gist = np.asarray(s["gist"])
    assert s["intatto"]
    dx = 1.0 / N
    righe: list[dict] = []
    for rum in rumori:
        ok = 0
        qw: list[float] = []
        qg: list[float] = []
        ug = 0
        for cid in ids:
            rec = rete.slot[cid]
            cp, vis = cue_parziale(rec["Fx"], frazione=frazione,
                                   tipo="blocco", rumore=float(rum),
                                   seed=cid * 13 + 1)
            sel = seleziona_prior(rete, cp, vis)
            ok += int(sel["best_id"] == cid)
            rec_w = rete.slot[sel["best_id"]]
            Fr_w = ricostruisci_associativo(cp, vis, rec_w["Fo"], dx,
                                            modo="residuo")
            Fr_g = ricostruisci_associativo(cp, vis, gist, dx,
                                            modo="residuo")
            qw.append(_q_regione(Fr_w, rec["Fx"], ~vis))
            qg.append(_q_regione(Fr_g, rec["Fx"], ~vis))
            ug += int(sel["margine"] < float(soglia_margine))
        righe.append({"rumore": float(rum),
                      "acc_winner": float(ok / max(1, len(ids))),
                      "q_winner_media": float(np.mean(qw)) if qw else 0.0,
                      "q_gist_media": float(np.mean(qg)) if qg else 0.0,
                      "fraz_gist": float(ug / max(1, len(ids)))})
    return righe


def salva_report_sonno(mix_righe: list[dict], rum_righe: list[dict],
                       out_dir: str | None = None,
                       N: int = 16, T: float = 0.05, seed: int = 7,
                       soglia_margine: float = 0.05) -> dict:
    """CSV + md + fig18 (sonno: winner vs gist su mix + rumore)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = OUT_DIR_DEFAULT
    _os.makedirs(out_dir, exist_ok=True)
    fp_csv = _os.path.join(out_dir, "fase16_sonno.csv")
    with open(fp_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["tipo", "alpha_rumore", "vincitore_acc", "margine",
                    "q_winner", "q_gist", "q_scelta"])
        for r in mix_righe:
            w.writerow(["mix", f"{r['alpha']:.2f}", r["vincitore"],
                        f"{r['margine']:.6f}",
                        f"{r['q_winner_vs_mix']:.6f}",
                        f"{r['q_gist_vs_mix']:.6f}",
                        f"{r['q_scelta_vs_mix']:.6f}"])
        for r in rum_righe:
            w.writerow(["rumore", f"{r['rumore']:.2f}",
                        f"{r['acc_winner']:.3f}", "",
                        f"{r['q_winner_media']:.6f}",
                        f"{r['q_gist_media']:.6f}", ""])
    fp_md = _os.path.join(out_dir, "fase16_sonno.md")
    with open(fp_md, "w") as f:
        f.write("# Fase 16 — Sonno (gist offline, slot intatti)\n\n")
        f.write(f"N={N} T={T} seed={seed} soglia_margine={soglia_margine}\n\n")
        f.write("Politica: se margine < soglia usa il gist, altrimenti il "
                "vincitore. Il sonno non scrive mai negli slot.\n\n")
        f.write("## Mix A+B\n\n")
        f.write("| alpha | vincitore | margine | q winner | q gist | q scelta |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in mix_righe:
            f.write(f"| {r['alpha']:.2f} | {r['vincitore']} | "
                    f"{r['margine']:.2e} | {r['q_winner_vs_mix']:.4f} | "
                    f"{r['q_gist_vs_mix']:.4f} | {r['q_scelta_vs_mix']:.4f} |\n")
        f.write("\n## Rumore\n\n")
        f.write("| rumore | acc winner | q winner | q gist | fraz gist |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rum_righe:
            f.write(f"| {r['rumore']:.2f} | {r['acc_winner']:.3f} | "
                    f"{r['q_winner_media']:.4f} | {r['q_gist_media']:.4f} | "
                    f"{r['fraz_gist']:.2f} |\n")
        f.write("\nLettura: il gist e' pooling sfocato; aiuta solo dove il "
                "winner e' ambiguo. Se q_gist <= q_winner ovunque, il sonno "
                "non risolve l'OOD (negativo onesto).\n")
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.5))
    al = [r["alpha"] for r in mix_righe]
    ax[0].plot(al, [r["q_winner_vs_mix"] for r in mix_righe], "o-",
               label="winner")
    ax[0].plot(al, [r["q_gist_vs_mix"] for r in mix_righe], "s-",
               label="gist")
    ax[0].plot(al, [r["q_scelta_vs_mix"] for r in mix_righe], "^--",
               label="scelta (soglia)")
    for x, r in zip(al, mix_righe):
        ax[0].text(x, r["q_scelta_vs_mix"], f" {r['vincitore']}", fontsize=9)
    ax[0].set_xlabel("alpha (1=puro A)")
    ax[0].set_ylabel("q vs mix")
    ax[0].set_title("A) Sonno su mix OOD")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot([r["rumore"] for r in rum_righe],
               [r["q_winner_media"] for r in rum_righe], "o-",
               label="winner")
    ax[1].plot([r["rumore"] for r in rum_righe],
               [r["q_gist_media"] for r in rum_righe], "s-",
               label="gist")
    ax[1].set_xlabel("rumore (x std)")
    ax[1].set_ylabel("q_mask media")
    ax[1].set_title("B) Sonno vs rumore")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"Sonno (N={N}, seed={seed}, soglia={soglia_margine})",
                 fontsize=13)
    fig.tight_layout()
    ffig = _os.path.join(out_dir, "fig18_sonno.png")
    fig.savefig(ffig, dpi=120)
    plt.close(fig)
    print(f"sonno: mix {len(mix_righe)} + rumore {len(rum_righe)} | fig {ffig}")
    return {"csv": fp_csv, "md": fp_md, "fig": ffig}


# ===========================================================================
# 2. DIMENTICARE APPOSTA
# ===========================================================================
def uso_proxy(ids: list[int]) -> dict[int, int]:
    """Proxy d'uso deterministico e skewato: uso = id % 3 (0 raro, 2 freq).

    Serve a distinguere LFU da FIFO senza simulare sessioni d'uso reali.
    Documentato come proxy, non come misura d'uso reale.
    """
    return {int(i): int(int(i) % 3) for i in ids}


def evici_slot(rete: ReteFrammento, k: int,
               politica: str = "eta",
               uso: dict[int, int] | None = None) -> list[int]:
    """Cancella k slot (mai sovrascrittura). Ritorna id evictati.

    - eta: id piu' piccoli prima (i piu' vecchi, apprendimento in ordine).
    - q: q_cert piu' bassa prima (i piu' deboli).
    - uso: conteggio piu' basso prima (i meno usati; default proxy id%3).
    Cancella anche la traccia cue_note; la cache MemoriaGF resta (solo
    cache, non verita'). Verifica il nucleo dopo (deve restare ok).
    """
    assert politica in ("eta", "q", "uso")
    k = max(0, int(k))
    ids = sorted(rete.slot.keys())
    if k <= 0 or not ids:
        return []
    k = min(k, len(ids))
    if politica == "eta":
        vitt = ids[:k]
    elif politica == "q":
        vitt = sorted(ids, key=lambda i: float(rete.slot[i]["q_cert"]))[:k]
    else:
        u = dict(uso) if uso is not None else uso_proxy(ids)
        vitt = sorted(ids, key=lambda i: (int(u.get(int(i), 0)), int(i)))[:k]
    for i in vitt:
        del rete.slot[int(i)]
        rete.cue_note.pop(int(i), None)
    v = rete.verifica_nucleo()
    assert v["ok"], "eviction ha corrotto il nucleo (non deve mai accadere)"
    return [int(i) for i in vitt]


def esegui_eviction_study(n: int = 40, N: int = 16, T: float = 0.05,
                          seed: int = 7,
                          k_list=(0, 10, 20, 30),
                          politiche=("eta", "q", "uso")) -> list[dict]:
    """Per ogni (politica, k): allena n, evici k, misura perdita.

    Metriche: copertura = rimasti/certificati, q media rimasti,
    memoria rimasta/risparmiata (4 campi float64/slot).
    """
    righe: list[dict] = []
    per_slot = PER_SLOT_CAMPI * N * N * 8
    for pol in politiche:
        for k in k_list:
            rete = ReteFrammento(N=N, T=T)
            for c in genera_cue(int(n), seed=seed):
                rete.impara(c)
            n_cert = len(rete.slot)
            qs_prima = [float(v["q_cert"]) for v in rete.slot.values()]
            q_prima = float(np.mean(qs_prima)) if qs_prima else 0.0
            ev = evici_slot(rete, int(k), politica=pol)
            qs = [float(v["q_cert"]) for v in rete.slot.values()]
            rim = len(rete.slot)
            righe.append({
                "politica": pol,
                "k_richiesti": int(k),
                "k_evictati": len(ev),
                "n_certificati": int(n_cert),
                "rimasti": int(rim),
                "copertura": float(rim / max(1, n_cert)),
                "q_media_prima": q_prima,
                "q_media_rimasti": float(np.mean(qs)) if qs else 0.0,
                "memoria_kb": float(rim * per_slot / 1024.0),
                "risparmio_kb": float(len(ev) * per_slot / 1024.0),
            })
    return righe


def salva_report_eviction(righe: list[dict], out_dir: str | None = None,
                          N: int = 16, T: float = 0.05, seed: int = 7,
                          n: int = 40) -> dict:
    """CSV + md + fig19 (copertura/Q/memoria vs k per politica)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = OUT_DIR_DEFAULT
    _os.makedirs(out_dir, exist_ok=True)
    fp_csv = _os.path.join(out_dir, "fase16_eviction.csv")
    with open(fp_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["politica", "k", "rimasti", "copertura",
                    "q_media_rimasti", "memoria_kb", "risparmio_kb"])
        for r in righe:
            w.writerow([r["politica"], r["k_evictati"], r["rimasti"],
                        f"{r['copertura']:.4f}",
                        f"{r['q_media_rimasti']:.6f}",
                        f"{r['memoria_kb']:.1f}",
                        f"{r['risparmio_kb']:.1f}"])
    fp_md = _os.path.join(out_dir, "fase16_eviction.md")
    with open(fp_md, "w") as f:
        f.write("# Fase 16 — Dimenticare apposta (eviction misurata)\n\n")
        f.write(f"n={n} N={N} T={T} seed={seed}\n\n")
        f.write("| politica | k | rimasti | copertura | Q rimasti | memoria | risparmio |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in righe:
            f.write(f"| {r['politica']} | {r['k_evictati']} | {r['rimasti']} "
                    f"| {r['copertura']:.3f} | {r['q_media_rimasti']:.4f} | "
                    f"{r['memoria_kb']:.0f} KB | {r['risparmio_kb']:.0f} KB |\n")
        f.write("\nEviction = cancellazione, mai sovrascrittura: i rimasti "
                "restano bit-identici (zero degrado sui non evictati).\n")
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))
    pols = sorted({r["politica"] for r in righe})
    for pol in pols:
        rr = [r for r in righe if r["politica"] == pol]
        ks = [r["k_evictati"] for r in rr]
        ax[0].plot(ks, [r["copertura"] for r in rr], "o-", label=pol)
        ax[1].plot(ks, [r["q_media_rimasti"] for r in rr], "o-", label=pol)
        ax[2].plot(ks, [r["memoria_kb"] for r in rr], "o-", label=pol)
    ax[0].set_xlabel("k evictati")
    ax[0].set_ylabel("copertura")
    ax[0].set_title("A) Copertura vs k")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("k evictati")
    ax[1].set_ylabel("Q media rimasti")
    ax[1].set_title("B) Qualita' dei rimasti (intatta)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[2].set_xlabel("k evictati")
    ax[2].set_ylabel("KB rimasti")
    ax[2].set_title("C) Memoria vs k (lineare)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.suptitle(f"Eviction (n={n}, N={N}, seed={seed})", fontsize=13)
    fig.tight_layout()
    ffig = _os.path.join(out_dir, "fig19_eviction.png")
    fig.savefig(ffig, dpi=120)
    plt.close(fig)
    print(f"eviction: {len(righe)} righe | fig {ffig}")
    return {"csv": fp_csv, "md": fp_md, "fig": ffig}


# ===========================================================================
# 3. METRICHE CL STANDARD (BWT/FWT a 2 task)
# ===========================================================================
def _nuovi_modelli(N: int, T: float, seed: int,
                   ewc_lam: float = 0.5) -> dict:
    return {
        "protetta": ReteFrammento(N=N, T=T),
        "ingenua": ReteIngenuaCondivisa(N=N, T=T),
        "replay": ReteReplay(N=N, T=T, replay_k=10, buffer_max=200,
                             seed=seed),
        "ewc": ReteEWC(N=N, T=T, lam=ewc_lam, eta=0.1),
    }


def _qA_ricorda(modello, nome: str, ids: list[int]) -> list[float]:
    out: list[float] = []
    for i in ids:
        if nome == "protetta":
            try:
                out.append(float(modello.richiama_ricalcola(int(i))["q_recall"]))
            except AssertionError:
                pass
        else:
            try:
                out.append(float(modello.richiama(int(i))["q_recall"]))
            except KeyError:
                pass
    return out


def misura_transfer(n_cert: int = 30, n_nuove: int = 60,
                    N: int = 16, T: float = 0.05, seed: int = 7,
                    ewc_lam: float = 0.5) -> dict:
    """BWT/FWT su shift A->B per 4 modelli (stesse sequenze, fair).

    - R_AA: Q su A dopo solo A; R_AB: Q su A dopo A+B; BWT = media(R_AB-R_AA).
    - R_B0: Q su B con modello addestrato solo su B; R_BB: Q su B dopo A+B;
      FWT = media(R_BB-R_B0) (quanto A aiuta/ostacola B).
    Protetta attesa BWT=FWT=0 (slot isolati); condivise con BWT<0.
    """
    A = genera_cue(int(n_cert), seed=seed, amp_range=(3.0, 8.0),
                   regione="sinistra")
    B = genera_cue(int(n_nuove), seed=seed + 1000, amp_range=(3.0, 8.0),
                   regione="destra")
    for k, c in enumerate(B):
        c["id"] = int(n_cert) + k

    # run AB: A poi B
    mod_ab = _nuovi_modelli(N, T, seed, ewc_lam)
    idA: list[int] = []
    for c in A:
        r = mod_ab["protetta"].impara(c)
        for m in ("ingenua", "replay", "ewc"):
            mod_ab[m].impara(c)
        if r["cert"]:
            idA.append(r["id"])
    raa = {m: _qA_ricorda(mod_ab[m], m, idA) for m in mod_ab}
    idB: list[int] = []
    for c in B:
        r = mod_ab["protetta"].impara(c)
        for m in ("ingenua", "replay", "ewc"):
            mod_ab[m].impara(c)
        if r["cert"]:
            idB.append(r["id"])
    rab = {m: _qA_ricorda(mod_ab[m], m, idA) for m in mod_ab}
    rbb = {m: _qA_ricorda(mod_ab[m], m, idB) for m in mod_ab}

    # run B-solo: stessi B su modelli freschi
    mod_b = _nuovi_modelli(N, T, seed, ewc_lam)
    for c in B:
        for m in mod_b:
            mod_b[m].impara(c)
    rb0 = {m: _qA_ricorda(mod_b[m], m, idB) for m in mod_b}

    per_modello: dict = {}
    for m in mod_ab:
        a0 = np.asarray(raa[m]) if raa[m] else np.zeros(0)
        a1 = np.asarray(rab[m]) if rab[m] else np.zeros(0)
        b1 = np.asarray(rbb[m]) if rbb[m] else np.zeros(0)
        b0 = np.asarray(rb0[m]) if rb0[m] else np.zeros(0)
        n = min(a0.size, a1.size)
        bwt = float(np.mean(a1[:n] - a0[:n])) if n else 0.0
        n2 = min(b1.size, b0.size)
        fwt = float(np.mean(b1[:n2] - b0[:n2])) if n2 else 0.0
        per_modello[m] = {
            "R_AA": float(np.mean(a0)) if a0.size else 0.0,
            "R_AB": float(np.mean(a1)) if a1.size else 0.0,
            "R_BB": float(np.mean(b1)) if b1.size else 0.0,
            "R_B0": float(np.mean(b0)) if b0.size else 0.0,
            "BWT": bwt,
            "FWT": fwt,
        }
    return {"n_A": len(idA), "n_B": len(idB), "N": N, "T": T, "seed": seed,
            "ewc_lam": float(ewc_lam), "per_modello": per_modello}


def salva_report_transfer(res: dict, out_dir: str | None = None) -> dict:
    """CSV + md + fig20 (barre BWT/FWT per modello)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = OUT_DIR_DEFAULT
    _os.makedirs(out_dir, exist_ok=True)
    pm = res["per_modello"]
    fp_csv = _os.path.join(out_dir, "fase16_transfer.csv")
    with open(fp_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["modello", "R_AA", "R_AB", "R_BB", "R_B0", "BWT", "FWT"])
        for m, v in pm.items():
            w.writerow([m, f"{v['R_AA']:.6f}", f"{v['R_AB']:.6f}",
                        f"{v['R_BB']:.6f}", f"{v['R_B0']:.6f}",
                        f"{v['BWT']:.6f}", f"{v['FWT']:.6f}"])
    fp_md = _os.path.join(out_dir, "fase16_transfer.md")
    with open(fp_md, "w") as f:
        f.write("# Fase 16 — Transfer standard (BWT/FWT, 2 task A->B)\n\n")
        f.write(f"A={res['n_A']} B={res['n_B']} N={res['N']} T={res['T']} "
                f"seed={res['seed']}\n\n")
        f.write("| modello | R_AA | R_AB | R_BB | R_B0 | BWT | FWT |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for m, v in pm.items():
            f.write(f"| {m} | {v['R_AA']:.4f} | {v['R_AB']:.4f} | "
                    f"{v['R_BB']:.4f} | {v['R_B0']:.4f} | {v['BWT']:.4f} | "
                    f"{v['FWT']:.4f} |\n")
        f.write("\nBWT<0 = forgetting; FWT>0 = A aiuta B. Protetta attesa "
                "BWT=FWT=0 (isolamento, non transfer).\n")
    nomi = list(pm.keys())
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax[0].bar(nomi, [pm[m]["BWT"] for m in nomi], color="#8b1e3f")
    ax[0].axhline(0.0, color="k", lw=1)
    ax[0].set_ylabel("BWT (Q_A dopo B - dopo A)")
    ax[0].set_title("A) Backward transfer (0 = zero forgetting)")
    ax[0].grid(alpha=0.3, axis="y")
    ax[1].bar(nomi, [pm[m]["FWT"] for m in nomi], color="#1f4e79")
    ax[1].axhline(0.0, color="k", lw=1)
    ax[1].set_ylabel("FWT (Q_B con A - senza A)")
    ax[1].set_title("B) Forward transfer (0 = isolamento)")
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle(f"Transfer A->B (A={res['n_A']}, B={res['n_B']})",
                 fontsize=13)
    fig.tight_layout()
    ffig = _os.path.join(out_dir, "fig20_transfer.png")
    fig.savefig(ffig, dpi=120)
    plt.close(fig)
    print(f"transfer: A {res['n_A']} B {res['n_B']} | fig {ffig}")
    return {"csv": fp_csv, "md": fp_md, "fig": ffig}


# ===========================================================================
# 4. RETRIEVAL REPAIR
# ===========================================================================
def split_vis(vis: np.ndarray, fraz_val: float = 0.2,
              seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Spezza l'osservato in fit/val (disgiunti, fit|val = vis).

    Il val e' tenuto da parte per la gara tra prior (mai l'occulto).
    Deterministico dato seed.
    """
    vis_a = np.asarray(vis, dtype=bool)
    idx = np.argwhere(vis_a)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    n_val = int(round(len(idx) * float(np.clip(fraz_val, 0.0, 0.5))))
    val = np.zeros_like(vis_a)
    if n_val > 0:
        for p in idx[:n_val]:
            val[tuple(p)] = True
    fit = vis_a & (~val)
    return fit, val


def topk_prior(rete: ReteFrammento, cue: np.ndarray, vis: np.ndarray,
               k: int = 3) -> list[tuple[float, int]]:
    """Top-k slot per MSE sul visibile (stesso score del baseline)."""
    cue_a = np.asarray(cue, dtype=float)
    vis_a = np.asarray(vis, dtype=bool)
    scored: list[tuple[float, int]] = []
    for cid, rec in rete.slot.items():
        d = cue_a[vis_a] - np.asarray(rec["Fx"], dtype=float)[vis_a]
        scored.append((float(np.mean(d ** 2)), int(cid)))
    scored.sort()
    return scored[:max(1, int(k))]


def seleziona_prior_validato(rete: ReteFrammento, cue: np.ndarray,
                             vis: np.ndarray, dx: float,
                             k: int = 3, fraz_val: float = 0.2,
                             seed: int = 0, T_rec: float = 5.0) -> dict:
    """Gara tra top-k prior in validazione (fit->val, mai occulto).

    Stesso split fit/val per tutti i candidati (fair). Per ogni candidato:
    ricostruisci dal fit col suo Fo, score = MSE(rec[val], cue[val]).
    Vince lo score minore. Margine = (second-best)/best.
    """
    cue_a = np.asarray(cue, dtype=float)
    vis_a = np.asarray(vis, dtype=bool)
    cand = topk_prior(rete, cue_a, vis_a, k=k)
    fit, val = split_vis(vis_a, fraz_val=fraz_val, seed=seed)
    if not val.any() or not fit.any():
        # fallback onesto: split impossibile -> baseline
        base = seleziona_prior(rete, cue_a, vis_a)
        return {"best_id": base["best_id"], "val_mse": float("nan"),
                "second_mse": float("nan"), "margine": 0.0,
                "n_candidati": len(cand), "fallback": True}
    cue_fit = np.where(fit, cue_a, 0.0)
    scores: list[tuple[float, int]] = []
    for _, cid in cand:
        rec = rete.slot[int(cid)]
        Fr = ricostruisci_associativo(cue_fit, fit, rec["Fo"], float(dx),
                                      T_rec=float(T_rec), modo="residuo")
        mse = float(np.mean((Fr[val] - cue_a[val]) ** 2))
        scores.append((mse, int(cid)))
    scores.sort()
    best_mse, best_id = scores[0]
    second = scores[1][0] if len(scores) > 1 else float("nan")
    marg = ((second - best_mse) / (best_mse + 1e-12)
            if len(scores) > 1 else 0.0)
    return {"best_id": int(best_id), "val_mse": float(best_mse),
            "second_mse": float(second), "margine": float(marg),
            "n_candidati": len(scores), "fallback": False}


def _downsample(a: np.ndarray, fattore: int = 2) -> np.ndarray:
    n = a.shape[0]
    f = int(fattore)
    assert n % f == 0, "downsample richiede N multiplo del fattore"
    return a.reshape(n // f, f, n // f, f).mean(axis=(1, 3))


def seleziona_prior_coarse_to_fine(rete: ReteFrammento, cue: np.ndarray,
                                   vis: np.ndarray, dx: float,
                                   fattore: int = 2, k: int = 3,
                                   fraz_val: float = 0.2, seed: int = 0,
                                   T_rec: float = 5.0) -> dict:
    """Scrematura grossolana + gara fine in validazione.

    Coarse: media a blocchi su cue/slot/Fx, MSE sul visibile coarse per
    i top-k. Fine: stessa gara di `seleziona_prior_validato` sui top-k.
    Se N non multiplo del fattore, fallback alla sola validazione fine.
    """
    N = rete.N
    f = int(fattore)
    if N % f != 0:
        r = seleziona_prior_validato(rete, cue, vis, dx, k=k,
                                     fraz_val=fraz_val, seed=seed,
                                     T_rec=T_rec)
        r["coarse"] = False
        return r
    cue_a = np.asarray(cue, dtype=float)
    vis_a = np.asarray(vis, dtype=bool)
    cc = _downsample(cue_a, f)
    cv = _downsample(vis_a.astype(float), f) > 0.5
    scored: list[tuple[float, int]] = []
    for cid, rec in rete.slot.items():
        Fx_c = _downsample(np.asarray(rec["Fx"], dtype=float), f)
        d = cc[cv] - Fx_c[cv]
        scored.append((float(np.mean(d ** 2)), int(cid)))
    scored.sort()
    top = [cid for _, cid in scored[:max(1, int(k))]]
    # gara fine ristretta ai top coarse
    fit, val = split_vis(vis_a, fraz_val=fraz_val, seed=seed)
    if not val.any() or not fit.any():
        base = seleziona_prior(rete, cue_a, vis_a)
        return {"best_id": base["best_id"], "val_mse": float("nan"),
                "second_mse": float("nan"), "margine": 0.0,
                "n_candidati": len(top), "fallback": True, "coarse": True}
    cue_fit = np.where(fit, cue_a, 0.0)
    scores: list[tuple[float, int]] = []
    for cid in top:
        rec = rete.slot[int(cid)]
        Fr = ricostruisci_associativo(cue_fit, fit, rec["Fo"], float(dx),
                                      T_rec=float(T_rec), modo="residuo")
        scores.append((float(np.mean((Fr[val] - cue_a[val]) ** 2)),
                       int(cid)))
    scores.sort()
    best_mse, best_id = scores[0]
    second = scores[1][0] if len(scores) > 1 else float("nan")
    marg = ((second - best_mse) / (best_mse + 1e-12)
            if len(scores) > 1 else 0.0)
    return {"best_id": int(best_id), "val_mse": float(best_mse),
            "second_mse": float(second), "margine": float(marg),
            "n_candidati": len(scores), "fallback": False, "coarse": True}


def valuta_retrieval_repair(N: int = 16, T: float = 0.05, seed: int = 7,
                            rumori=(0.0, 0.5, 1.0, 2.0),
                            frazione: float = 0.5, k: int = 3,
                            fraz_val: float = 0.2) -> list[dict]:
    """Baseline vs validato vs coarse-to-fine al crescere del rumore."""
    cue = genera_cue(8, seed=seed, amp_range=(3.0, 8.0))
    rete = ReteFrammento(N=N, T=T)
    for c in cue:
        rete.impara(c)
    ids = sorted(rete.slot.keys())
    dx = 1.0 / N
    righe: list[dict] = []
    for rum in rumori:
        ok_b = ok_v = ok_c = 0
        mv: list[float] = []
        for cid in ids:
            rec = rete.slot[cid]
            cp, vis = cue_parziale(rec["Fx"], frazione=frazione,
                                   tipo="blocco", rumore=float(rum),
                                   seed=cid * 13 + 1)
            b = seleziona_prior(rete, cp, vis)
            v = seleziona_prior_validato(rete, cp, vis, dx, k=k,
                                         fraz_val=fraz_val,
                                         seed=cid * 131 + 7)
            cc = seleziona_prior_coarse_to_fine(rete, cp, vis, dx,
                                                fattore=2, k=k,
                                                fraz_val=fraz_val,
                                                seed=cid * 131 + 7)
            ok_b += int(b["best_id"] == cid)
            ok_v += int(v["best_id"] == cid)
            ok_c += int(cc["best_id"] == cid)
            mv.append(float(v["margine"]))
        n = max(1, len(ids))
        righe.append({"rumore": float(rum),
                      "acc_base": float(ok_b / n),
                      "acc_validato": float(ok_v / n),
                      "acc_coarse": float(ok_c / n),
                      "margine_val_medio": float(np.mean(mv)) if mv else 0.0})
    return righe


def salva_report_retrieval(righe: list[dict], out_dir: str | None = None,
                           N: int = 16, T: float = 0.05, seed: int = 7,
                           k: int = 3, fraz_val: float = 0.2) -> dict:
    """CSV + md + fig21 (accuracy baseline vs repair vs rumore)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = OUT_DIR_DEFAULT
    _os.makedirs(out_dir, exist_ok=True)
    fp_csv = _os.path.join(out_dir, "fase16_retrieval.csv")
    with open(fp_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["rumore", "acc_base", "acc_validato", "acc_coarse",
                    "margine_val_medio"])
        for r in righe:
            w.writerow([f"{r['rumore']:.2f}", f"{r['acc_base']:.3f}",
                        f"{r['acc_validato']:.3f}", f"{r['acc_coarse']:.3f}",
                        f"{r['margine_val_medio']:.6f}"])
    fp_md = _os.path.join(out_dir, "fase16_retrieval.md")
    with open(fp_md, "w") as f:
        f.write("# Fase 16 — Retrieval repair (validazione + coarse-to-fine)\n\n")
        f.write(f"N={N} T={T} seed={seed} k={k} fraz_val={fraz_val}\n\n")
        f.write("| rumore | base | validato | coarse-to-fine |\n")
        f.write("|---|---|---|---|\n")
        for r in righe:
            f.write(f"| {r['rumore']:.2f} | {r['acc_base']:.3f} | "
                    f"{r['acc_validato']:.3f} | {r['acc_coarse']:.3f} |\n")
        f.write("\nGara tra prior solo sul visibile (fit->val): mai "
                "l'occulto. Se validato <= base, il negativo OOD resta "
                "(onesto).\n")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    xs = [r["rumore"] for r in righe]
    ax.plot(xs, [r["acc_base"] for r in righe], "o-", label="base (MSE vis)")
    ax.plot(xs, [r["acc_validato"] for r in righe], "s-",
            label="validato (fit->val)")
    ax.plot(xs, [r["acc_coarse"] for r in righe], "^--",
            label="coarse-to-fine")
    ax.set_xlabel("rumore (x std)")
    ax.set_ylabel("accuratezza retrieval")
    ax.set_title(f"Retrieval repair (N={N}, k={k})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ffig = _os.path.join(out_dir, "fig21_retrieval.png")
    fig.savefig(ffig, dpi=120)
    plt.close(fig)
    print(f"retrieval: {len(righe)} rumori | fig {ffig}")
    return {"csv": fp_csv, "md": fp_md, "fig": ffig}
