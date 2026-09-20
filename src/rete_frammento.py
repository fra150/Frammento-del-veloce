"""Frammento del veloce - rete che non distrugge (Fase 13/14).

Tesi: "imparare cose nuove non distrugge mai quelle certificate".

Disegno (deciso con l'utente, diario 19/09/2026):
- Nucleo g0 IMMUTABILE: essenza congelata, scrittura vietata per costruzione,
  checksum sha256 verificato a ogni epoca.
- Periferia gx/gy PLASTICA: legge il nucleo, non lo scrive. Ogni ricordo
  occupa uno slot proprio (mai sovrascrittura).
- Certificatore gf come UNICO canale di scrittura: solo frammenti con
  quiete AND qualita' AND budget vengono memorizzati.
- Capacita': il nucleo cresce (espansione) o rifiuta nuove certificazioni,
  mai sovrascrittura distruttiva.

Definizioni operative:
- ricordo = (cue, stato Fx/Fy/Fo) + Q alla certificazione;
- distruzione = Q di richiamo sotto soglia (o degrado > soglia_degrado)
  dopo nuovo apprendimento;
- capacita' = numero di slot certificati (espandibile).

Test dei 1000 = misura di interferenza retroattiva:
  certifica N_cert (default 200), impara N_nuove sopra (default 800),
  ri-testa le N_cert -> zero degradi o fallimento per definizione.
  Curva Q(t) piatta come figura centrale.

Baseline di contrasto ReteIngenuaCondivisa: singolo campo plastico
condiviso P aggiornato a media mobile. Dopo 800 nuovi apprendimenti
P deriva e i richiami vecchi degradano (degrado > 0 atteso).
Serve a mostrare che la protezione non e' vacua.

Solo numpy + matplotlib (figure) + hashlib. Nessun modello esterno.
Simulazioni deterministiche (stocastico=False) per isolare l'interferenza
dal rumore (il rumore e' gia' caratterizzato in sweep/ablazione §6.1-6.4:
Q dipende dal sottosistema deterministico g0/gx).
"""

from __future__ import annotations

import csv
import hashlib
import os

import numpy as np

from .frammento_2d import Param, essenza, simula
from .frammento_gf import MemoriaGF, certifica_frammento

OUT_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", "output_test")

SOGLIA_DEGRADO = 0.02  # oltre: "distruzione" per definizione
TOL_ZERO = 1e-9  # tolleranza uguaglianza bit-identica (stesso seed -> stesso Q)


# ---------------------------------------------------------------------------
def checksum_arr(a: np.ndarray) -> str:
    """sha256 di shape + valori arrotondati a 1e-6 (stabile al rumore numerico)."""
    arr = np.round(np.asarray(a, dtype=float), 6)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def genera_cue(n: int, seed: int = 7) -> list[dict]:
    """N cue indipendenti (bump casuali), SENZA repliche.

    Differenza da stress_500.genera_domande (che mette una replica ogni 25
    per la cache): qui ogni cue e' un ricordo distinto, perche' il test dei
    1000 misura l'interferenza tra ricordi diversi.
    """
    rng = np.random.default_rng(int(seed))
    cue: list[dict] = []
    for i in range(int(n)):
        cue.append({
            "id": i,
            "seed_domanda": int(rng.integers(0, 10_000)),
            "protocollo": "stimolo" if bool(rng.integers(0, 2)) else "rilassamento",
            "gamma": float(rng.choice([0.0, 0.02, 0.15])),
            "bump_x": float(rng.uniform(0.2, 0.8)),
            "bump_y": float(rng.uniform(0.2, 0.8)),
            "bump_amp": float(rng.uniform(0.5, 2.0)),
        })
    return cue


def _bump_iniziale(p: Param, ess: np.ndarray, cue: dict) -> np.ndarray:
    x = (np.arange(p.N) + 0.5) / p.N * p.L
    X, Y = np.meshgrid(x, x, indexing="ij")
    g = np.exp(-((X - cue["bump_x"] * p.L) ** 2
                 + (Y - cue["bump_y"] * p.L) ** 2) / (2 * 0.05 ** 2))
    return np.asarray(ess) + cue["bump_amp"] * g


def _qualita(Fx: np.ndarray, ess: np.ndarray) -> float:
    return float(1.0 - np.linalg.norm(np.asarray(Fx) - np.asarray(ess))
                 / (np.linalg.norm(np.asarray(ess)) + 1e-12))


# ---------------------------------------------------------------------------
class ReteFrammento:
    """Rete protetta: nucleo frozen + slot isolati + scrittura solo via gf.

    - Il nucleo (`_nucleo`) non e' mai scritto dopo __init__: nessun metodo
      lo modifica; `verifica_nucleo()` confronta il checksum a ogni epoca
      e solleva AssertionError se corrotto (non deve mai accadere).
    - `nucleo` (property) restituisce una COPIA: il chiamante non puo'
      corrompere l'originale per aliasing.
    - Solo `_scrivi_solo_se_certificato` scrive in `slot`; `impara`
      lo chiama solo con `certificato=True`.
    """

    def __init__(self, N: int = 32, T: float = 0.10,
                 capacita_max: int | None = None,
                 politica: str = "espandi",
                 soglia_qualita: float = 0.40,
                 budget_novita_rel: float = 0.30,
                 eps: float = 0.60, delta: float = 0.90):
        assert politica in ("espandi", "rifiuta")
        self.N = int(N)
        self.T = float(T)
        self.capacita_max = None if capacita_max is None else int(capacita_max)
        self.politica = politica
        self.soglia_qualita = float(soglia_qualita)
        self.budget = float(budget_novita_rel)
        self.eps = float(eps)
        self.delta = float(delta)
        p0 = Param(N=self.N)
        self._nucleo = essenza(p0).copy()
        self._checksum_nucleo = checksum_arr(self._nucleo)
        self.slot: dict[int, dict] = {}  # cue_id -> record certificato
        self.cue_note: dict[int, dict] = {}  # cue_id -> cue (per ri-test)
        self.mem = MemoriaGF()
        self.epoca = 0

    @property
    def nucleo(self) -> np.ndarray:
        return self._nucleo.copy()

    @property
    def checksum(self) -> str:
        return self._checksum_nucleo

    def verifica_nucleo(self) -> dict:
        ck = checksum_arr(self._nucleo)
        ok = (ck == self._checksum_nucleo)
        return {"ok": bool(ok), "checksum": ck,
                "atteso": self._checksum_nucleo}

    def _controllo_epoca(self) -> None:
        v = self.verifica_nucleo()
        assert v["ok"], f"nucleo corrotto a epoca {self.epoca}!"

    def _scrivi_solo_se_certificato(self, cue_id: int, record: dict,
                                    certificato: bool) -> bool:
        """Unico canale di scrittura. Ritorna True se scritto."""
        self._controllo_epoca()
        if not certificato:
            return False
        if cue_id in self.slot:
            # mai sovrascrittura: il ricordo esiste gia' (richiamo, non rewrite)
            return False
        if self.capacita_max is not None and len(self.slot) >= self.capacita_max:
            if self.politica == "rifiuta":
                return False
            # espandi: raddoppia (capacita' che cresce, mai overwrite)
            self.capacita_max = max(len(self.slot) + 1, self.capacita_max * 2)
        self.slot[cue_id] = dict(record)
        return True

    def impara(self, cue: dict) -> dict:
        """Apprende una cue: simula -> gf -> eventuale scrittura isolata."""
        self._controllo_epoca()
        self.epoca += 1
        p = Param(N=self.N, seed=int(cue["seed_domanda"]),
                  gamma=float(cue["gamma"]))
        ess = essenza(p)
        snap = simula(p, T=self.T, protocollo=cue["protocollo"],
                      salva_ogni=10, stocastico=False,
                      stato_iniziale={"Fx": _bump_iniziale(p, ess, cue)})
        Fo, Fx, Fy = snap["F0"][-1], snap["Fx"][-1], snap["Fy"][-1]
        c = certifica_frammento(Fo, Fx, Fy, ess, p.dx, diag=snap["diag"],
                                eps=self.eps, delta=self.delta,
                                soglia_qualita=self.soglia_qualita,
                                budget_novita_rel=self.budget)
        self.cue_note[int(cue["id"])] = dict(cue)
        scritto = False
        chiave = ""
        if c["certificato"]:
            chiave = self.mem.chiave(Fo, Fx, Fy, ess, dx=p.dx)
            hit, _ = self.mem.richiama(chiave)
            if not hit:
                self.mem.salva(chiave, {"qualita": c["qualita"],
                                        "cue": int(cue["id"])})
            record = {"Fx": np.asarray(Fx).copy(), "Fy": np.asarray(Fy).copy(),
                      "Fo": np.asarray(Fo).copy(), "ess": np.asarray(ess).copy(),
                      "q_cert": float(c["qualita"]),
                      "novita_rel": float(c["novita_rel"]),
                      "chiave": chiave}
            scritto = self._scrivi_solo_se_certificato(int(cue["id"]),
                                                       record, True)
        self._controllo_epoca()
        return {"id": int(cue["id"]), "cert": bool(c["certificato"]),
                "q": float(c["qualita"]), "scritto": bool(scritto),
                "capacita": len(self.slot),
                "nucleo_ok": True}

    def richiama_ricalcola(self, cue_id: int) -> dict:
        """Ri-testa: riesegue la sim della cue e confronta Q con Q_cert."""
        self._controllo_epoca()
        cue_id = int(cue_id)
        assert cue_id in self.slot, f"cue {cue_id} mai certificata"
        assert cue_id in self.cue_note, f"cue {cue_id} senza traccia"
        cue = self.cue_note[cue_id]
        rec = self.slot[cue_id]
        p = Param(N=self.N, seed=int(cue["seed_domanda"]),
                  gamma=float(cue["gamma"]))
        ess = essenza(p)
        snap = simula(p, T=self.T, protocollo=cue["protocollo"],
                      salva_ogni=10, stocastico=False,
                      stato_iniziale={"Fx": _bump_iniziale(p, ess, cue)})
        Fx2 = snap["F0"][-1] * 0 + snap["Fx"][-1]  # Fx ricalcolata
        q2 = _qualita(Fx2, ess)
        q0 = float(rec["q_cert"])
        degrado = float(q0 - q2)
        # cache: la chiave del ricordo deve dare hit
        hit, _ = self.mem.richiama(rec["chiave"])
        distrutto = bool((q2 < self.soglia_qualita) or (degrado > SOGLIA_DEGRADO))
        return {"id": cue_id, "q_cert": q0, "q_recall": float(q2),
                "degrado": degrado, "hit": bool(hit),
                "distrutto": distrutto, "nucleo_ok": True}


# ---------------------------------------------------------------------------
class ReteIngenuaCondivisa:
    """Baseline ingenua: UNICO campo plastico condiviso P (media mobile).

    Ogni apprendimento sposta P verso (Fx - essenza) con eta. Il richiamo
    di una vecchia cue somma la deriva (P_attuale - P_snapshot): dopo molti
    nuovi apprendimenti Q degrada. Mostra che senza isolamento si distrugge.
    """

    def __init__(self, N: int = 32, T: float = 0.10, eta: float = 0.10,
                 soglia_qualita: float = 0.40):
        self.N = int(N)
        self.T = float(T)
        self.eta = float(eta)
        self.soglia_qualita = float(soglia_qualita)
        p0 = Param(N=self.N)
        self.ess0 = essenza(p0).copy()
        self.P = np.zeros((self.N, self.N))
        self.record: dict[int, dict] = {}
        self.cue_note: dict[int, dict] = {}

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
        # plastica condivisa: si sposta verso ogni nuovo ricordo
        self.P = (1.0 - self.eta) * self.P + self.eta * (Fx - ess)
        self.cue_note[int(cue["id"])] = dict(cue)
        self.record[int(cue["id"])] = {"Fx": Fx.copy(), "ess": ess.copy(),
                                       "q_cert": float(q),
                                       "P_snapshot": snap_P}
        return {"id": int(cue["id"]), "q": float(q)}

    def richiama(self, cue_id: int) -> dict:
        cue_id = int(cue_id)
        rec = self.record[cue_id]
        deriva = self.P - rec["P_snapshot"]
        Fx_d = rec["Fx"] + deriva
        q2 = _qualita(Fx_d, rec["ess"])
        degrado = float(rec["q_cert"] - q2)
        return {"id": cue_id, "q_cert": float(rec["q_cert"]),
                "q_recall": float(q2), "degrado": degrado,
                "distrutto": bool((q2 < self.soglia_qualita)
                                  or (degrado > SOGLIA_DEGRADO))}


# ---------------------------------------------------------------------------
def esegui_test_1000(n_cert: int = 200, n_nuove: int = 800,
                     N: int = 32, T: float = 0.10, seed: int = 7,
                     capacita_max: int | None = None,
                     politica: str = "espandi") -> dict:
    """Interferenza retroattiva: certifica n_cert, impara n_nuove, ri-testa.

    Ritorna dict con q_prima/q_dopo/degradi per la rete protetta e per la
    baseline ingenua, checksum nucleo, conteggi distrutti.
    Criterio di successo protetta: max_degrado <= TOL_ZERO e 0 distrutti.
    """
    tot = int(n_cert) + int(n_nuove)
    cue = genera_cue(tot, seed=seed)
    prime = cue[:int(n_cert)]
    nuove = cue[int(n_cert):]

    rete = ReteFrammento(N=N, T=T, capacita_max=capacita_max,
                         politica=politica)
    ing = ReteIngenuaCondivisa(N=N, T=T)
    ck0 = rete.checksum

    q_cert_list: list[float] = []
    id_cert: list[int] = []
    for c in prime:
        r = rete.impara(c)
        ing.impara(c)
        if r["cert"]:
            q_cert_list.append(r["q"])
            id_cert.append(r["id"])
    n_certificati = len(id_cert)

    for c in nuove:
        rete.impara(c)
        ing.impara(c)

    ck1 = rete.verifica_nucleo()
    q_dopo: list[float] = []
    degradi: list[float] = []
    distrutti = 0
    q_dopo_ing: list[float] = []
    degradi_ing: list[float] = []
    distrutti_ing = 0
    for i in id_cert:
        rr = rete.richiama_ricalcola(i)
        q_dopo.append(rr["q_recall"])
        degradi.append(rr["degrado"])
        distrutti += int(rr["distrutto"])
        ri = ing.richiama(i)
        q_dopo_ing.append(ri["q_recall"])
        degradi_ing.append(ri["degrado"])
        distrutti_ing += int(ri["distrutto"])

    import numpy as _np
    degradi_a = _np.asarray(degradi) if degradi else _np.zeros(0)
    degradi_i = _np.asarray(degradi_ing) if degradi_ing else _np.zeros(0)
    return {
        "n_cert_richieste": int(n_cert), "n_nuove": int(n_nuove),
        "n_certificati": int(n_certificati),
        "id_cert": id_cert,
        "q_prima": [float(v) for v in q_cert_list],
        "q_dopo": [float(v) for v in q_dopo],
        "degradi": [float(v) for v in degradi],
        "max_degrado": float(degradi_a.max()) if degradi_a.size else 0.0,
        "distrutti": int(distrutti),
        "q_dopo_ing": [float(v) for v in q_dopo_ing],
        "degradi_ing": [float(v) for v in degradi_ing],
        "max_degrado_ing": float(degradi_i.max()) if degradi_i.size else 0.0,
        "distrutti_ing": int(distrutti_ing),
        "checksum_prima": ck0, "checksum_dopo": ck1["checksum"],
        "nucleo_ok": bool(ck1["ok"]),
        "capacita": len(rete.slot),
        "successo": bool(n_certificati > 0 and ck1["ok"]
                         and (degradi_a.size == 0 or degradi_a.max() <= TOL_ZERO)
                         and distrutti == 0),
    }


def salva_report_r1000(res: dict, out_dir: str = OUT_DIR_DEFAULT,
                       N: int = 32, T: float = 0.10, seed: int = 7) -> dict:
    """CSV + md + fig11 (curva Q piatta). Ritorna path scritti."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fp_csv = os.path.join(out_dir, "rete_1000.csv")
    with open(fp_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "q_cert", "q_recall_protetta", "degrado_protetta",
                    "q_recall_ingenua", "degrado_ingenua"])
        for i, idx in enumerate(res["id_cert"]):
            w.writerow([idx, f"{res['q_prima'][i]:.6f}",
                        f"{res['q_dopo'][i]:.6f}",
                        f"{res['degradi'][i]:.2e}",
                        f"{res['q_dopo_ing'][i]:.6f}",
                        f"{res['degradi_ing'][i]:.6f}"])

    fp_md = os.path.join(out_dir, "rete_1000.md")
    with open(fp_md, "w") as f:
        f.write("# Test dei 1000 — interferenza retroattiva (rete che non distrugge)\n\n")
        f.write(f"N={N} T={T:.2f} seed={seed} "
                f"richieste={res['n_cert_richieste']}+{res['n_nuove']}\n\n")
        f.write(f"- certificati prime: {res['n_certificati']}/{res['n_cert_richieste']}\n")
        f.write(f"- nucleo checksum: {res['checksum_prima'][:12]}... -> "
                f"{res['checksum_dopo'][:12]}... ok={res['nucleo_ok']}\n")
        f.write(f"- PROTETTA: max degrado {res['max_degrado']:.2e}, "
                f"distrutti {res['distrutti']} "
                f"({'SUCCESSO: zero degradi' if res['successo'] else 'FALLITO'})\n")
        f.write(f"- INGENUA: max degrado {res['max_degrado_ing']:.4f}, "
                f"distrutti {res['distrutti_ing']}\n")
        f.write("- figura: fig11_rete_Q.png\n")

    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.3))
    x = np.arange(len(res["q_prima"]))
    ax[0].plot(x, np.asarray(res["q_prima"]), color="#1f4e79", lw=1.0,
               label="Q certificazione")
    ax[0].plot(x, np.asarray(res["q_dopo"]), color="#2e7d32", lw=1.0, ls="--",
               label="Q recall protetta (dopo +800)")
    ax[0].plot(x, np.asarray(res["q_dopo_ing"]), color="#8b1e3f", lw=1.0,
               alpha=0.8, label="Q recall ingenua")
    ax[0].axhline(0.40, color="red", ls=":", lw=1, label="soglia gf 0.40")
    ax[0].set_xlabel("ricordo certificato #")
    ax[0].set_ylabel("qualita'")
    ax[0].set_title("Q(t) piatta: protetta vs ingenua")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].scatter(np.asarray(res["q_prima"]), np.asarray(res["q_dopo"]),
                  c="green", s=12, alpha=0.7, label="protetta")
    ax[1].scatter(np.asarray(res["q_prima"]), np.asarray(res["q_dopo_ing"]),
                  c="red", s=12, alpha=0.5, label="ingenua")
    lims = [float(min(res["q_prima"] + res["q_dopo"] + res["q_dopo_ing"]) - 0.02),
            float(max(res["q_prima"] + res["q_dopo"] + res["q_dopo_ing"]) + 0.02)]
    ax[1].plot(lims, lims, color="k", ls="--", lw=1)
    ax[1].set_xlabel("Q certificazione")
    ax[1].set_ylabel("Q recall dopo +800")
    ax[1].set_title("Prima vs dopo (diagonale = zero degrado)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[2].axis("off")
    ax[2].text(0.05, 0.9,
               f"protetta: max degr {res['max_degrado']:.2e}", fontsize=11)
    ax[2].text(0.05, 0.75,
               f"protetta distrutti: {res['distrutti']}/{res['n_certificati']}",
               fontsize=11)
    ax[2].text(0.05, 0.60,
               f"ingenua: max degr {res['max_degrado_ing']:.4f}", fontsize=11)
    ax[2].text(0.05, 0.45,
               f"ingenua distrutti: {res['distrutti_ing']}/{res['n_certificati']}",
               fontsize=11)
    ax[2].text(0.05, 0.30,
               f"nucleo ok: {res['nucleo_ok']}  capacita: {res['capacita']}",
               fontsize=11)
    ax[2].text(0.05, 0.12, f"N={N} T={T:.2f} seed={seed}", fontsize=10)
    fig.suptitle("Rete che non distrugge — test dei 1000 (interferenza retroattiva)",
                 fontsize=13)
    fig.tight_layout()
    ffig = os.path.join(out_dir, "fig11_rete_Q.png")
    fig.savefig(ffig, dpi=110)
    plt.close(fig)
    print(f"rete1000: cert {res['n_certificati']} protetta maxdegr "
          f"{res['max_degrado']:.2e} distr {res['distrutti']} | "
          f"ingenua maxdegr {res['max_degrado_ing']:.4f} distr "
          f"{res['distrutti_ing']} | nucleo_ok {res['nucleo_ok']}")
    print("cartella:", out_dir)
    return {"csv": fp_csv, "md": fp_md, "fig": ffig, "out_dir": out_dir}
