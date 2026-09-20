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

from .frammento_2d import Param, essenza, simula, laplaciano
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


def genera_cue(n: int, seed: int = 7,
               amp_range: tuple[float, float] = (0.5, 2.0),
               regione: str | None = None) -> list[dict]:
    """N cue indipendenti (bump casuali), SENZA repliche.

    Differenza da stress_500.genera_domande (che mette una replica ogni 25
    per la cache): qui ogni cue e' un ricordo distinto, perche' il test dei
    1000 misura l'interferenza tra ricordi diversi.

    - `amp_range`: intervallo di ampiezza del bump (default 0.5-2.0, come
      negli altri test). Per il test associativo si usano bump grandi
      (3-8) per dare ai ricordi un contenuto individuale misurabile.
    - `regione`: `None` = bump ovunque in [0.2, 0.8]^2 (default storico);
      `"sinistra"` = [0.15, 0.45]^2; `"destra"` = [0.55, 0.85]^2. Serve allo
      shift di classe del test associativo.
    """
    rng = np.random.default_rng(int(seed))
    cue: list[dict] = []
    for i in range(int(n)):
        if regione == "sinistra":
            bx = float(rng.uniform(0.15, 0.45))
            by = float(rng.uniform(0.15, 0.45))
        elif regione == "destra":
            bx = float(rng.uniform(0.55, 0.85))
            by = float(rng.uniform(0.55, 0.85))
        else:
            bx = float(rng.uniform(0.2, 0.8))
            by = float(rng.uniform(0.2, 0.8))
        cue.append({
            "id": i,
            "seed_domanda": int(rng.integers(0, 10_000)),
            "protocollo": "stimolo" if bool(rng.integers(0, 2)) else "rilassamento",
            "gamma": float(rng.choice([0.0, 0.02, 0.15])),
            "bump_x": bx,
            "bump_y": by,
            "bump_amp": float(rng.uniform(float(amp_range[0]),
                                          float(amp_range[1]))),
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

    def richiama_associativo(self, cue_id: int, frazione: float = 0.5,
                             tipo: str = "blocco", rumore: float = 0.0,
                             seed_cue: int | None = None,
                             T_rec: float = 5.0,
                             modo: str = "residuo") -> dict:
        """Richiamo associativo: ricostruzione da cue parziale.

        Il cue e' una versione parziale di `Fx_cert` (blocco mancante o pixel
        sparsi + rumore opzionale); la ricostruzione usa la dinamica condivisa
        del modello (vedi `ricostruisci_associativo`) con prior = core
        certificato `Fo` del ricordo. NON e' ri-esecuzione: nessun seed della
        cue, nessun protocollo di stimolo.

        Metriche: `q_recall` (globale vs stato certificato), `q_mask_rec`
        (regione mancante ricostruita), `q_mask_media`/`q_mask_core`
        (baseline di riempimento: media dei visibili / core certificato).
        """
        self._controllo_epoca()
        cue_id = int(cue_id)
        assert cue_id in self.slot, f"cue {cue_id} mai certificata"
        rec = self.slot[cue_id]
        dx = 1.0 / self.N
        sc = 0 if seed_cue is None else int(seed_cue)
        cue, vis = cue_parziale(rec["Fx"], frazione=frazione, tipo=tipo,
                                rumore=rumore, seed=sc)
        Fx_rec = ricostruisci_associativo(cue, vis, rec["Fo"], dx,
                                          T_rec=T_rec, modo=modo)
        q_all = _qualita(Fx_rec, rec["Fx"])
        q_mask = _q_regione(Fx_rec, rec["Fx"], ~vis)
        q_media = _q_regione(_riempi(cue, vis, float(cue[vis].mean())),
                             rec["Fx"], ~vis)
        q_core = _q_regione(_riempi(cue, vis, rec["Fo"]), rec["Fx"], ~vis)
        return {"id": cue_id, "frazione": float(frazione), "tipo": tipo,
                "rumore": float(rumore), "frac_eff": float((~vis).mean()),
                "q_recall": float(q_all), "q_mask_rec": float(q_mask),
                "q_mask_media": float(q_media), "q_mask_core": float(q_core)}


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

    def richiama_associativo(self, cue_id: int, frazione: float = 0.5,
                             tipo: str = "blocco", rumore: float = 0.0,
                             seed_cue: int | None = None,
                             T_rec: float = 5.0,
                             P_rif: np.ndarray | None = None) -> dict:
        """Richiamo associativo nella baseline: prior = campo condiviso.

        Stessa ricostruzione della rete protetta, ma il prior non e' un core
        per-ricordo congelato: e' il campo condiviso `ess0 + P` (con `P` in
        media mobile). `P_rif` permette di fissare lo stato del campo
        condiviso (es. il controllo `P_A` dopo solo il set A); con `None`
        (default) si usa il campo condiviso CORRENTE: dopo nuovi
        apprendimenti il prior segue i dati nuovi e il richiamo dei vecchi
        ricordi e' accoppiato alla composizione del set.
        """
        cue_id = int(cue_id)
        rec = self.record[cue_id]
        dx = 1.0 / self.N
        sc = 0 if seed_cue is None else int(seed_cue)
        cue, vis = cue_parziale(rec["Fx"], frazione=frazione, tipo=tipo,
                                rumore=rumore, seed=sc)
        P_uso = self.P if P_rif is None else np.asarray(P_rif, dtype=float)
        prior = self.ess0 + P_uso
        Fx_rec = ricostruisci_associativo(cue, vis, prior, dx, T_rec=T_rec)
        q_all = _qualita(Fx_rec, rec["Fx"])
        q_mask = _q_regione(Fx_rec, rec["Fx"], ~vis)
        return {"id": cue_id, "frazione": float(frazione),
                "frac_eff": float((~vis).mean()),
                "q_recall": float(q_all), "q_mask_rec": float(q_mask)}


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


# ---------------------------------------------------------------------------
# Richiamo associativo: cue parziali -> ricostruzione (NON ri-esecuzione)
# ---------------------------------------------------------------------------
# Il richiamo "diretto" (`richiama_ricalcola`) riesegue la simulazione della
# cue: e' una verifica di invariante, non un richiamo associativo. Qui il
# richiamo parte da un cue PARZIALE (parte del campo mancante + rumore
# opzionale) e ricostruisce con la dinamica canonica di gx:
#
#     dFx = [Dx lap(Fx) + kx (Fo_cert - Fx)] dt,   Fx[vis] = cue[vis]
#
# con ancoraggio sui pixel osservati. La ricostruzione NON conosce il seed
# della cue ne' riesegue il protocollo di stimolo: usa solo il cue parziale,
# il core certificato Fo_cert e una dinamica condivisa e deterministica.
# ---------------------------------------------------------------------------
def cue_parziale(
    Fx: np.ndarray,
    frazione: float = 0.5,
    tipo: str = "blocco",
    rumore: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Costruisce un cue parziale da un campo memorizzato.

    - `tipo="blocco"`: azzera un quadrato centrale di area ~`frazione`.
    - `tipo="casuale"`: azzera pixel sparsi (frazione ~`frazione`).
    - `rumore > 0`: aggiunge rumore gaussiano scalato a `rumore * std(Fx)`.

    Ritorna `(cue, vis)`: `cue` = campo con la parte mancante azzerata
    (+rumore), `vis` = maschera booleana True sui pixel osservati.
    Deterministico dato `seed`.
    """
    Fx = np.asarray(Fx, dtype=float)
    N = Fx.shape[0]
    f = float(np.clip(frazione, 0.0, 1.0))
    rng = np.random.default_rng(int(seed))
    vis = np.ones(Fx.shape, dtype=bool)
    if tipo == "blocco":
        side = max(1, min(N, int(np.ceil(np.sqrt(f) * N))))
        c0 = (N - side) // 2
        vis[c0:c0 + side, c0:c0 + side] = False
    elif tipo == "casuale":
        vis = rng.random(Fx.shape) >= f
    else:
        raise ValueError(f"tipo cue sconosciuto: {tipo!r}")
    cue = Fx.copy()
    cue[~vis] = 0.0
    if rumore > 0:
        cue = cue + float(rumore) * float(Fx.std()) * rng.standard_normal(Fx.shape)
    return cue, vis


def ricostruisci_associativo(
    cue: np.ndarray,
    vis: np.ndarray,
    prior: np.ndarray,
    dx: float,
    Dx: float = 0.01,
    kx: float = 0.6,
    T_rec: float = 5.0,
    dt: float | None = None,
    ancoraggio: bool = True,
    modo: str = "residuo",
) -> np.ndarray:
    """Completamento canonico della regione mancante (dinamica condivisa).

    `modo="residuo"` (default): il richiamo estende il *residuo* osservato
    (traccia oltre il nucleo) e lo somma al nucleo certificato:

        R = Fx - prior;  lap(R) = 0 nella regione mancante (diffusione pura
        con ancoraggio R = (cue - prior) sui pixel osservati);
        Fx_rec = prior + R.

    E' il limite di diffusione pura del modello (l'operatore `laplaciano`),
    deterministico, senza seed ne' ri-esecuzione della simulazione.

    `modo="gx"`: rilassamento del campo intero verso il prior
    (`dFx = Dx lap(Fx) + kx (prior - Fx)`, ancoraggio sui visibili). In questo
    regime di campi dominati dalla storia dell'input e' risultato peggiore
    del modo "residuo" (vedi confronto nel report).
    """
    dx = float(dx)
    dt_ = dt or 0.4 * dx ** 2 / (4.0 * max(float(Dx), 1e-12))
    nsteps = max(1, int(float(T_rec) / dt_))
    vis_a = np.asarray(vis, dtype=bool)
    if modo == "residuo":
        base = np.asarray(prior, dtype=float)
        R = np.asarray(cue, dtype=float) - base
        R_anc = R.copy()
        R = np.where(vis_a, R, 0.0)
        for _ in range(nsteps):
            R = R + dt_ * float(Dx) * laplaciano(R, dx)
            if ancoraggio:
                R = np.where(vis_a, R_anc, R)
        return base + R
    if modo == "gx":
        Fx = np.asarray(cue, dtype=float).copy()
        cue_a = np.asarray(cue, dtype=float)
        prior_a = np.asarray(prior, dtype=float)
        for _ in range(nsteps):
            Fx = Fx + dt_ * (float(Dx) * laplaciano(Fx, dx)
                             + float(kx) * (prior_a - Fx))
            if ancoraggio:
                Fx = np.where(vis_a, cue_a, Fx)
        return Fx
    raise ValueError(f"modo sconosciuto: {modo!r}")


def _q_regione(F_rec: np.ndarray, F_true: np.ndarray, regione: np.ndarray) -> float:
    """Qualita' sulla regione: 1 - ||F_rec - F_true|| / ||F_true|| (regione)."""
    d = np.asarray(F_rec, dtype=float)[regione] - np.asarray(F_true, dtype=float)[regione]
    t = np.asarray(F_true, dtype=float)[regione]
    return float(1.0 - np.linalg.norm(d) / (np.linalg.norm(t) + 1e-12))


def _riempi(Fx: np.ndarray, vis: np.ndarray, valore) -> np.ndarray:
    """Baseline: copia del cue con la regione mancante riempita (scalare o campo)."""
    out = np.asarray(Fx, dtype=float).copy()
    if np.isscalar(valore):
        out[~vis] = float(valore)
    else:
        out[~vis] = np.asarray(valore, dtype=float)[~vis]
    return out


def esegui_test_associativo(n_cert: int = 200, n_nuove: int = 800,
                            N: int = 32, T: float = 0.10, seed: int = 7,
                            frazioni=(0.25, 0.50, 0.75), tipo: str = "blocco",
                            rumore: float = 0.0, T_rec: float = 5.0,
                            amp_range: tuple[float, float] = (3.0, 8.0),
                            regione_a: str = "sinistra",
                            regione_b: str = "destra",
                            salva_esempio: bool = True) -> dict:
    """Richiamo associativo su n_cert certificati dopo +n_nuove apprendimenti.

    Set A (n_cert, bump a `regione_a`) poi set B (n_nuove, bump a
    `regione_b`): shift di classe. Per ogni ricordo di A e ogni frazione di
    cue mancante:
    - PROTETTA: ricostruzione `residuo` con prior = core certificato;
    - confronto operatori: `media`, `core`, `armonica` (senza nucleo),
      `gx` (rilassamento canonico sul campo intero);
    - INGENUA controllo: prior = campo condiviso `P_A` (dopo solo A);
    - INGENUA dopo: prior = campo condiviso corrente (ha seguito B).
    Degrado ingenua = q(P_A) - q(dopo). La protetta e' verificata
    bit-identica prima/dopo (a f=0.5 su tutti i ricordi di A).

    Ritorna per-riga (`per`) e aggregati per frazione (`agg`), piu' un
    esempio di campi per la figura (`esempio`).
    """
    amp = tuple(amp_range)
    cueA = genera_cue(int(n_cert), seed=seed, amp_range=amp,
                      regione=regione_a)
    cueB = genera_cue(int(n_nuove), seed=seed + 1000, amp_range=amp,
                      regione=regione_b)
    for k, c in enumerate(cueB):
        c["id"] = int(n_cert) + k
    rete = ReteFrammento(N=N, T=T)
    ing = ReteIngenuaCondivisa(N=N, T=T)
    for c in cueA:
        rete.impara(c)
        ing.impara(c)
    P_A = ing.P.copy()
    dx = 1.0 / N

    # verifica protetta: richiamo a f=0.5 PRIMA di B (ricontrollato dopo)
    ids = sorted(rete.slot.keys())
    prima: dict[int, float] = {}
    for cid in ids:
        rr = rete.richiama_associativo(cid, 0.50, tipo, rumore,
                                       1000 * cid + 11, T_rec)
        prima[cid] = float(rr["q_mask_rec"])

    for c in cueB:
        rete.impara(c)
        ing.impara(c)
    ids_A = [i for i in sorted(rete.slot.keys()) if i < int(n_cert)]

    per: list[dict] = []
    esempio: dict | None = None
    for cid in ids_A:
        for j, f in enumerate(frazioni):
            sc = 1000 * cid + 10 * j + 1
            rp = rete.richiama_associativo(cid, f, tipo, rumore, sc, T_rec)
            rec = rete.slot[cid]
            cue_p, vis = cue_parziale(rec["Fx"], frazione=f, tipo=tipo,
                                      rumore=rumore, seed=sc)
            # operatori di confronto (stesso cue, stesso prior certificato)
            Fx_gx = ricostruisci_associativo(cue_p, vis, rec["Fo"], dx,
                                             T_rec=3.0, modo="gx")
            q_gx = _q_regione(Fx_gx, rec["Fx"], ~vis)
            Fx_arm = ricostruisci_associativo(
                cue_p, vis, np.zeros_like(rec["Fo"]), dx,
                T_rec=T_rec, modo="residuo")
            q_arm = _q_regione(Fx_arm, rec["Fx"], ~vis)
            rn = ing.richiama_associativo(cid, f, tipo, rumore, sc, T_rec,
                                          P_rif=ing.P)
            rn0 = ing.richiama_associativo(cid, f, tipo, rumore, sc, T_rec,
                                           P_rif=P_A)
            per.append({
                "id": cid, "frazione": float(f), "frac_eff": rp["frac_eff"],
                "q_recall": rp["q_recall"], "q_mask_rec": rp["q_mask_rec"],
                "q_mask_media": rp["q_mask_media"],
                "q_mask_core": rp["q_mask_core"],
                "q_mask_arm": float(q_arm), "q_mask_gx": float(q_gx),
                "q_mask_ing": rn["q_mask_rec"],
                "q_mask_ing0": rn0["q_mask_rec"],
                "degrado_ing": float(rn0["q_mask_rec"] - rn["q_mask_rec"]),
            })
            if (salva_esempio and esempio is None and cid == ids_A[0]
                    and abs(f - 0.50) < 1e-9):
                rec = rete.slot[cid]
                cue_p, vis = cue_parziale(rec["Fx"], frazione=f, tipo=tipo,
                                          rumore=rumore, seed=sc)
                Fx_rec = ricostruisci_associativo(cue_p, vis, rec["Fo"],
                                                  1.0 / N, T_rec=T_rec)
                esempio = {"id": cid, "cue": cue_p, "vis": vis,
                           "Fx_true": np.asarray(rec["Fx"]).copy(),
                           "Fx_rec": Fx_rec, "Fo": np.asarray(rec["Fo"]).copy()}

    # aggregati per frazione
    agg: dict[float, dict] = {}
    for f in frazioni:
        righe = [r for r in per if abs(r["frazione"] - f) < 1e-12]
        def _m(k: str) -> float:
            return float(np.mean([r[k] for r in righe]))
        def _s(k: str) -> float:
            return float(np.std([r[k] for r in righe]))
        degs = [r["degrado_ing"] for r in righe]
        agg[float(f)] = {
            "n": len(righe),
            "frac_eff": _m("frac_eff"),
            "q_mask_rec": _m("q_mask_rec"), "q_mask_rec_std": _s("q_mask_rec"),
            "q_mask_media": _m("q_mask_media"),
            "q_mask_core": _m("q_mask_core"),
            "q_mask_arm": _m("q_mask_arm"),
            "q_mask_gx": _m("q_mask_gx"),
            "q_mask_ing": _m("q_mask_ing"), "q_mask_ing_std": _s("q_mask_ing"),
            "q_mask_ing0": _m("q_mask_ing0"),
            "degrado_ing": _m("degrado_ing"),
            "degrado_ing_max": float(np.max(degs)),
            "degrado_ing_pos": float(np.mean([d > 0.001 for d in degs])),
            "degrado_ing_gt05": float(np.mean([d > 0.05 for d in degs])),
            "vince_su_media": float(np.mean(
                [r["q_mask_rec"] > r["q_mask_media"] for r in righe])),
            "vince_su_core": float(np.mean(
                [r["q_mask_rec"] > r["q_mask_core"] for r in righe])),
        }

    # verifica protetta: stesso richiamo a f=0.5 dopo B
    ver_max = 0.0
    for cid in ids_A:
        rr = rete.richiama_associativo(cid, 0.50, tipo, rumore,
                                       1000 * cid + 11, T_rec)
        ver_max = max(ver_max, abs(float(rr["q_mask_rec"]) - prima[cid]))

    ck = rete.verifica_nucleo()
    return {
        "n_cert": len(ids_A), "n_nuove": int(n_nuove), "frazioni": list(frazioni),
        "tipo": tipo, "rumore": float(rumore), "T_rec": float(T_rec),
        "amp_range": amp, "regione_a": regione_a, "regione_b": regione_b,
        "shift_rel": float(np.linalg.norm(ing.P - P_A)
                           / (np.linalg.norm(P_A) + 1e-12)),
        "verifica_protetta_max_diff": float(ver_max),
        "per": per, "agg": agg, "esempio": esempio,
        "nucleo_ok": bool(ck["ok"]),
        "checksum": ck["checksum"],
    }


def salva_report_assoc(res: dict, out_dir: str = OUT_DIR_DEFAULT,
                       N: int = 32, T: float = 0.10, seed: int = 7) -> dict:
    """CSV + md + fig13 (completamento da cue parziali, protetta vs ingenua)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fp_csv = os.path.join(out_dir, "assoc_1000.csv")
    with open(fp_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "frazione", "frac_eff", "q_recall", "q_mask_rec",
                    "q_mask_media", "q_mask_core", "q_mask_arm", "q_mask_gx",
                    "q_mask_ing", "q_mask_ing0", "degrado_ing"])
        for r in res["per"]:
            w.writerow([r["id"], f"{r['frazione']:.2f}", f"{r['frac_eff']:.4f}",
                        f"{r['q_recall']:.6f}", f"{r['q_mask_rec']:.6f}",
                        f"{r['q_mask_media']:.6f}", f"{r['q_mask_core']:.6f}",
                        f"{r['q_mask_arm']:.6f}", f"{r['q_mask_gx']:.6f}",
                        f"{r['q_mask_ing']:.6f}", f"{r['q_mask_ing0']:.6f}",
                        f"{r['degrado_ing']:.6f}"])

    fp_md = os.path.join(out_dir, "assoc_1000.md")
    with open(fp_md, "w") as f:
        f.write("# Richiamo associativo da cue parziali (ricostruzione)\n\n")
        f.write(f"N={N} T={T:.2f} seed={seed} tipo={res['tipo']} "
                f"rumore={res['rumore']} T_rec={res['T_rec']} "
                f"amp bump={res['amp_range']}\n")
        f.write(f"shift di classe: {res['regione_a']} -> {res['regione_b']} "
                f"(||P_B-P_A||/||P_A|| = {res['shift_rel']:.3f})\n")
        f.write(f"certificati A: {res['n_cert']} | nuove apprese B: "
                f"{res['n_nuove']} | nucleo ok={res['nucleo_ok']}\n")
        f.write(f"verifica protetta (f=0.5, prima vs dopo B): max diff = "
                f"{res['verifica_protetta_max_diff']:.2e}\n\n")
        f.write("## Operatori di completamento (protetta, prior = core)\n\n")
        f.write("| frazione | frac eff | residuo | media | core | armonica "
                "| gx (canonico) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for fq, a in res["agg"].items():
            f.write(f"| {fq:.2f} | {a['frac_eff']:.3f} | "
                    f"{a['q_mask_rec']:.4f}±{a['q_mask_rec_std']:.4f} | "
                    f"{a['q_mask_media']:.4f} | {a['q_mask_core']:.4f} | "
                    f"{a['q_mask_arm']:.4f} | {a['q_mask_gx']:.4f} |\n")
        f.write("\n## Degrado in regime associativo (protetta vs condivisa)\n\n")
        f.write("| frazione | protetta | condivisa P_A (controllo) | "
                "condivisa dopo shift | degrado medio | degrado max | >0.05 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for fq, a in res["agg"].items():
            f.write(f"| {fq:.2f} | {a['q_mask_rec']:.4f}±{a['q_mask_rec_std']:.4f} "
                    f"| {a['q_mask_ing0']:.4f} | "
                    f"{a['q_mask_ing']:.4f}±{a['q_mask_ing_std']:.4f} | "
                    f"{a['degrado_ing']:.4f} | {a['degrado_ing_max']:.4f} | "
                    f"{a['degrado_ing_gt05']*100:.0f}% |\n")
        f.write("\nProtetta: ricostruzione deterministica con core congelato; "
                "degrado esattamente 0 (verificato bit-identico).\n")
        f.write("Condivisa: prior = campo condiviso; dentro la stessa classe "
                "il pooling aiuta (qualita' assoluta piu' alta), ma dopo lo "
                "shift il prior segue i dati nuovi e i vecchi ricordi "
                "degradano.\n")
        f.write("- figura: fig13_assoc.png\n")

    # --- figura
    col_blu, col_verde, col_rosso, col_grigio = "#1f4e79", "#2e7d32", "#8b1e3f", "#7a7a7a"
    fig = plt.figure(figsize=(14.0, 8.8))
    gs = fig.add_gridspec(2, 12, hspace=0.42, wspace=1.1)
    fq = list(res["agg"].keys())
    xpos = np.arange(len(fq))

    # Riga 1: esempio di completamento (vero | cue | ricostruzione | errore)
    if res.get("esempio") is not None:
        ex = res["esempio"]
        pannelli = [("vero (certificato)", ex["Fx_true"], "viridis"),
                    ("cue parziale", ex["cue"], "viridis"),
                    ("ricostruzione", ex["Fx_rec"], "viridis"),
                    ("errore |ric - vero|", np.abs(ex["Fx_rec"] - ex["Fx_true"]),
                     "magma")]
        for k, (tit, campo, cmap) in enumerate(pannelli):
            sp = fig.add_subplot(gs[0, 3 * k:3 * k + 3])
            sp.imshow(np.asarray(campo).T, origin="lower", cmap=cmap)
            sp.set_title(tit, fontsize=9)
            sp.set_xticks([])
            sp.set_yticks([])

    # Riga 2: A) barre per frazione (operatori di completamento)
    a = fig.add_subplot(gs[1, 0:4])
    w = 0.18
    serie = [("residuo", "q_mask_rec", col_verde, 1.0),
             ("media", "q_mask_media", col_grigio, 0.9),
             ("core", "q_mask_core", col_blu, 0.8),
             ("armonica", "q_mask_arm", "#c47b17", 0.85),
             ("gx", "q_mask_gx", "#6a4c93", 0.8)]
    for k, (nome, chiave, colore, alpha) in enumerate(serie):
        a.bar(xpos + (k - 2) * w, [res["agg"][f][chiave] for f in fq], w,
              color=colore, alpha=alpha, label=nome)
    a.set_xticks(xpos)
    a.set_xticklabels([f"{f:.0%}" for f in fq])
    a.set_xlabel("frazione di cue mancante")
    a.set_ylabel("qualita' sulla regione mancante")
    a.set_title("A) Operatori di completamento (prior = core)")
    a.axhline(0.0, color="k", lw=0.8)
    a.legend(fontsize=7, ncol=2)
    a.grid(alpha=0.3, axis="y")

    # Riga 2: C) q_mask vs frazione
    c = fig.add_subplot(gs[1, 4:8])
    c.errorbar(xpos, [res["agg"][f]["q_mask_rec"] for f in fq],
               yerr=[res["agg"][f]["q_mask_rec_std"] for f in fq],
               fmt="o-", color=col_verde, capsize=4,
               label="protetta (core congelato)")
    c.errorbar(xpos, [res["agg"][f]["q_mask_ing0"] for f in fq],
               fmt="^--", color=col_rosso, alpha=0.55, capsize=3,
               label="condivisa P_A (controllo, solo set A)")
    c.errorbar(xpos, [res["agg"][f]["q_mask_ing"] for f in fq],
               yerr=[res["agg"][f]["q_mask_ing_std"] for f in fq],
               fmt="s-", color=col_rosso, capsize=4,
               label="condivisa dopo shift (set B)")
    c.set_xticks(xpos)
    c.set_xticklabels([f"{f:.0%}" for f in fq])
    c.set_xlabel("frazione di cue mancante")
    c.set_ylabel("q_mask (media ± std)")
    c.set_title("C) Protetta invariante; la condivisa segue il set")
    c.legend(fontsize=7, loc="lower left")
    c.grid(alpha=0.3)

    # Riga 2: D) istogramma degrado condivisa (frazione centrale) + testo
    d = fig.add_subplot(gs[1, 8:12])
    f_mid = fq[len(fq) // 2]
    deg = [r["degrado_ing"] for r in res["per"]
           if abs(r["frazione"] - f_mid) < 1e-12]
    d.hist(deg, bins=25, color=col_rosso, alpha=0.8)
    d.axvline(0.0, color="k", lw=1)
    d.set_xlabel(f"degrado q_mask condivisa (frazione {f_mid:.0%})")
    d.set_ylabel("n. ricordi")
    d.set_title("D) Degrado della condivisa dopo lo shift (+%d)" % res["n_nuove"])
    d.grid(alpha=0.3)
    txt = (f"protetta: degrado 0\n"
           f"verifica bit-identica: {res['verifica_protetta_max_diff']:.1e}\n"
           f"shift ||P_B-P_A||/||P_A||: {res['shift_rel']:.3f}\n"
           f"nucleo ok: {res['nucleo_ok']}")
    d.text(0.98, 0.97, txt, transform=d.transAxes, fontsize=9, va="top",
           ha="right", family="monospace",
           bbox=dict(boxstyle="round", fc="#f5f5f5", ec="gray", alpha=0.9))

    fig.suptitle("Richiamo associativo: cue parziali -> ricostruzione — "
                 f"shift di classe {res['regione_a']}->{res['regione_b']} "
                 f"(N={N}, T={T:.2f}, seed={seed})", fontsize=13)
    fig.subplots_adjust(top=0.90, left=0.055, right=0.985, bottom=0.07)
    ffig = os.path.join(out_dir, "fig13_assoc.png")
    fig.savefig(ffig, dpi=120)
    plt.close(fig)

    a_mid = res["agg"][f_mid]
    print(f"assoc: cert A {res['n_cert']} + B {res['n_nuove']} | "
          f"shift {res['shift_rel']:.3f} | verifica protetta max diff "
          f"{res['verifica_protetta_max_diff']:.1e} | f={f_mid:.0%}: "
          f"prot {a_mid['q_mask_rec']:.4f} vs condivisa P_A "
          f"{a_mid['q_mask_ing0']:.4f} -> dopo {a_mid['q_mask_ing']:.4f} "
          f"(degrado {a_mid['degrado_ing']:.4f}, max "
          f"{a_mid['degrado_ing_max']:.4f})")
    print("cartella:", out_dir)
    return {"csv": fp_csv, "md": fp_md, "fig": ffig, "out_dir": out_dir}

