"""Frammento del veloce - modulo GF (quiete, certificazione, memoria).

Livello GF (garanzia del frammento): verifica che lo stato operativo `Fx`
sia in quiete rispetto all'essenza `Fo`/di riferimento, certifica il
frammento solo se la quiete e' attiva e la qualita' supera la soglia,
propone micro-correzioni leggere e memorizza i certificati in cache.

Tre livelli richiamati (vedi `frammento_2d` / `frammento_1d`):
    g0 : essenza invariante (qui `Fo` / `essenza`)
    gx : operativita' vincolata (qui `Fx`)
    gy : novita' controllata (qui `Fy`)

Regola fondamentale: MAI certificare se la quiete e' inattiva.
La correzione `correggi_micro_errori` NON certifica: propone solo una
media pesata leggera verso `Fo`, da rivalutare con `verifica_quiete` e
`certifica_frammento`.

Dipendenze: solo numpy + hashlib (stdlib).
"""

from __future__ import annotations

import hashlib

import numpy as np

EPS = 1e-12

__all__ = [
    "EPS",
    "verifica_quiete",
    "certifica_frammento",
    "correggi_micro_errori",
    "MemoriaGF",
    "diagnostica_gf",
]


# ---------------------------------------------------------------------------
# Utilita' interne
# ---------------------------------------------------------------------------
def _fattore_volume(Fo: np.ndarray, dx) -> float:
    """Fattore di volume per la massa: dx in 1D, dx**2 in 2D.

    Accetta sia `dx` scalare (passo di griglia) sia sequenza `(dx, dy)` o
    `(dx, dy, ...)`: in tal caso il volume e' il prodotto dei passi.
    Per array a `ndim` dimensioni con `dx` scalare il fattore e' `dx**ndim`,
    quindi 1D -> `dx`, 2D -> `dx**2`. Se `dx` e' None, ritorna 1.0
    (massa = semplice somma, utile per test adimensionali).
    """
    if dx is None:
        return 1.0
    arr = np.asarray(Fo, dtype=float)
    ndim = arr.ndim if arr.ndim > 0 else 1
    if isinstance(dx, (tuple, list, np.ndarray)):
        parti = [float(v) for v in np.asarray(dx, dtype=float).ravel()]
        if len(parti) == 0:
            return 1.0
        if len(parti) == 1:
            d = parti[0]
            return d if ndim <= 1 else d ** ndim
        volume = 1.0
        for v in parti:
            volume *= v
        return float(volume)
    d = float(dx)
    return d if ndim <= 1 else d ** ndim


def _arrotonda(a: np.ndarray) -> np.ndarray:
    """Arrotonda a 1e-6 per stabilizzare l'hash contro il rumore numerico."""
    return np.round(np.asarray(a, dtype=float), 6)


# ---------------------------------------------------------------------------
# 1. Verifica di quiete
# ---------------------------------------------------------------------------
def verifica_quiete(
    Fo: np.ndarray,
    Fx: np.ndarray,
    essenza: np.ndarray,
    dx,
    eps: float = 0.60,
    delta: float = 0.90,
    Fy: np.ndarray | None = None,
    V_hist: np.ndarray | list | None = None,
    budget_novita_rel: float = 0.25,
) -> dict:
    """Verifica che l'operativita' sia in quiete rispetto all'essenza.

    NOTA TARATURA (Fase gf, 19/09/2026): Fo diffonde per natura (D0=0.05,
    R_g0=0) quindi ||Fo-essenza|| cresce a ~0.3-0.8 su T=0.30 anche in
    regime sano (massa 1.0->1.0, dV/dt<=0 100%). Soglie strette 0.05
    renderebbero la quiete impossibile per disegno. Default pratici:
    eps=0.60 (Fx≈Fo) e delta=0.90 (Fo≈essenza larga), con V monotona e
    budget novita' come cancelli veri.

    Errori relativi (norme discrete, il passo `dx` si elide nel rapporto
    ed e' tenuto solo per coerenza API e invalidazione cache):

        err_fo_ess = ||Fo - essenza|| / (||essenza|| + EPS)
        err_fx_fo  = ||Fx - Fo|| / (||Fo|| + EPS)

    quiete attiva se `(err_fo_ess < delta) and (err_fx_fo < eps)`.

    Vincoli aggiuntivi (ognuno puo' solo disattivare, mai attivare):

    - se `Fy` e' fornito: `novita_rel = ||Fy|| / (||essenza|| + EPS)`;
      se supera `budget_novita_rel` (default 0.25, come `novelty_budget`
      del 1D) la quiete e' disattivata;
    - se `V_hist` ha >= 2 punti: la frazione di passi con `dV <= 0`
      (soglia 1e-9, come in `riepilogo` del 2D) deve essere >= 0.9,
      altrimenti la quiete e' disattivata. Con < 2 punti il controllo
      e' saltato (nessuna pendenza stimabile, vedi `derivata_numerica`).

    Ritorna dict con `attivo`, `err_fx_fo`, `err_fo_ess`, `novita_rel`,
    `fraz_dV_nonpos` (None se non valutabile) e `motivo` leggibile.
    """
    Fo_a = np.asarray(Fo, dtype=float)
    Fx_a = np.asarray(Fx, dtype=float)
    ess_a = np.asarray(essenza, dtype=float)

    n_ess = float(np.linalg.norm(ess_a))
    n_fo = float(np.linalg.norm(Fo_a))
    err_fo_ess = float(np.linalg.norm(Fo_a - ess_a) / (n_ess + EPS))
    err_fx_fo = float(np.linalg.norm(Fx_a - Fo_a) / (n_fo + EPS))

    attivo = bool((err_fo_ess < delta) and (err_fx_fo < eps))
    motivi: list[str] = []
    if err_fo_ess >= delta:
        motivi.append(f"deriva essenza err_fo_ess={err_fo_ess:.4f}>=delta={delta}")
    if err_fx_fo >= eps:
        motivi.append(f"operativita' non allineata err_fx_fo={err_fx_fo:.4f}>=eps={eps}")

    if Fy is None:
        novita_rel = 0.0
    else:
        Fy_a = np.asarray(Fy, dtype=float)
        novita_rel = float(np.linalg.norm(Fy_a) / (n_ess + EPS))
        if novita_rel > budget_novita_rel:
            attivo = False
            motivi.append(
                f"novita' eccessiva novita_rel={novita_rel:.4f}>budget={budget_novita_rel}"
            )

    fraz_dv: float | None = None
    if V_hist is not None:
        V = np.asarray(V_hist, dtype=float).ravel()
        if V.size >= 2:
            dV = np.diff(V)
            fraz_dv = float(np.mean(dV <= 1e-9))
            if fraz_dv < 0.9:
                attivo = False
                motivi.append(
                    f"Lyapunov non monotona fraz_dV<=0={fraz_dv:.3f}<0.9"
                )

    if attivo:
        motivo = (
            f"quiete attiva: err_fx_fo={err_fx_fo:.4f}<{eps}, "
            f"err_fo_ess={err_fo_ess:.4f}<{delta}, novita_rel={novita_rel:.4f}"
        )
    else:
        motivo = "quiete inattiva: " + ("; ".join(motivi) if motivi else "soglie superate")

    return {
        "attivo": bool(attivo),
        "err_fx_fo": float(err_fx_fo),
        "err_fo_ess": float(err_fo_ess),
        "novita_rel": float(novita_rel),
        "fraz_dV_nonpos": None if fraz_dv is None else float(fraz_dv),
        "motivo": motivo,
    }


# ---------------------------------------------------------------------------
# 2. Certificazione del frammento
# ---------------------------------------------------------------------------
def certifica_frammento(
    Fo: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    essenza: np.ndarray,
    dx,
    diag: dict | None = None,
    Fin: np.ndarray | None = None,
    eps: float = 0.60,
    delta: float = 0.90,
    soglia_qualita: float = 0.40,
    budget_novita_rel: float = 0.30,
) -> dict:
    """Certifica il frammento solo se quiete attiva, qualita' e budget ok.

    - Chiama `verifica_quiete` (con `V_hist` estratta da `diag["V"]` se
      presente, altrimenti None; `Fy` passata per il budget di novita').
    - `qualita = 1 - ||Fx - essenza|| / (||essenza|| + EPS)` (come in
      `metriche` del 2D, ma sul singolo stato finale).
    - `massa_fo = sum(Fo) * volume`, con volume = `dx` in 1D, `dx**2`
      in 2D (vedi `_fattore_volume`; accetta anche `dx` sequenza).
    - `novita = ||Fy||` (norma discreta); `novita_rel` dalla quiete.
    - `adattamento`, solo se `Fin` fornito:
      `1 - ||Fx - Fin|| / (||essenza - Fin|| + EPS)` (stessa forma di
      `_adattamento` in `studi.py`; None se `Fin` e' None).

    `certificato = True` solo se quiete attiva AND qualita' >= soglia
    AND novita_rel <= budget. MAI certificato se quiete = False.

    Ritorna dict con `certificato`, `qualita`, `adattamento`, `quiete`
    (dict di `verifica_quiete`), `motivo`, piu' `massa_fo`, `novita`,
    `novita_rel` per diagnostica.
    """
    V_hist = None
    if isinstance(diag, dict) and "V" in diag and diag["V"] is not None:
        V_hist = diag["V"]

    quiete = verifica_quiete(
        Fo,
        Fx,
        essenza,
        dx,
        eps=eps,
        delta=delta,
        Fy=Fy,
        V_hist=V_hist,
        budget_novita_rel=budget_novita_rel,
    )

    Fo_a = np.asarray(Fo, dtype=float)
    Fx_a = np.asarray(Fx, dtype=float)
    Fy_a = np.asarray(Fy, dtype=float)
    ess_a = np.asarray(essenza, dtype=float)

    n_ess = float(np.linalg.norm(ess_a))
    qualita = float(1.0 - np.linalg.norm(Fx_a - ess_a) / (n_ess + EPS))
    massa_fo = float(np.sum(Fo_a) * _fattore_volume(Fo_a, dx))
    novita = float(np.linalg.norm(Fy_a))
    novita_rel = float(quiete["novita_rel"])

    if Fin is None:
        adattamento = None
    else:
        Fin_a = np.asarray(Fin, dtype=float)
        adattamento = float(
            1.0 - np.linalg.norm(Fx_a - Fin_a) / (np.linalg.norm(ess_a - Fin_a) + EPS)
        )

    certificato = bool(
        quiete["attivo"]
        and (qualita >= soglia_qualita)
        and (novita_rel <= budget_novita_rel)
    )

    if not quiete["attivo"]:
        motivo = f"non certificato: {quiete['motivo']}"
    elif qualita < soglia_qualita:
        motivo = (
            f"non certificato: qualita={qualita:.4f}<soglia={soglia_qualita}"
        )
    elif novita_rel > budget_novita_rel:
        motivo = (
            f"non certificato: novita_rel={novita_rel:.4f}>budget={budget_novita_rel}"
        )
    else:
        motivo = (
            f"certificato: quiete attiva, qualita={qualita:.4f}>={soglia_qualita}, "
            f"novita_rel={novita_rel:.4f}<={budget_novita_rel}"
        )

    return {
        "certificato": bool(certificato),
        "qualita": float(qualita),
        "adattamento": None if adattamento is None else float(adattamento),
        "quiete": quiete,
        "motivo": motivo,
        "massa_fo": float(massa_fo),
        "novita": float(novita),
        "novita_rel": float(novita_rel),
    }


# ---------------------------------------------------------------------------
# 3. Micro-correzione (non certificante)
# ---------------------------------------------------------------------------
def correggi_micro_errori(
    Fx: np.ndarray, Fo: np.ndarray, fattore: float = 0.1
) -> np.ndarray:
    """Propone una correzione leggera verso l'essenza operativa.

    `Fx_corr = (1 - fattore) * Fx + fattore * Fo` (default fattore=0.1).

    NON certifica nulla: e' solo un passo di richiamo parziale, da
    rivalutare con `verifica_quiete` / `certifica_frammento` prima di
    qualsiasi uso come ricordo valido. `fattore` atteso in [0, 1];
    valori fuori intervallo sono ritagliati (clip) per sicurezza.
    """
    f = float(np.clip(float(fattore), 0.0, 1.0))
    Fx_a = np.asarray(Fx, dtype=float)
    Fo_a = np.asarray(Fo, dtype=float)
    return (1.0 - f) * Fx_a + f * Fo_a


# ---------------------------------------------------------------------------
# 4. Memoria GF: cache certificata a costo di recall ~0
# ---------------------------------------------------------------------------
class MemoriaGF:
    """Cache dei frammenti certificati (recall a costo ~0).

    La chiave e' lo sha256 di shape + contenuto (arrotondato a 1e-6) di
    `Fo`, `Fx`, `Fy`, `essenza` e di `dx`: `richiama` non ricalcola nulla,
    restituisce solo il payload memorizzato. Invalidazione implicita:
    qualsiasi cambio di valori, forma/N o passo `dx` produce una chiave
    diversa (miss), quindi nessun certificato obsoleto e' mai riusato.
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._salvataggi: int = 0

    def chiave(self, *args, dx=None, **kwargs) -> str:
        """Hash sha256 di shape + bytes arrotondati a 1e-6 (+ `dx`).

        Forme ammesse (flessibile per compatibilita'):

        - `chiave(Fo, Fx, Fy, essenza[, dx])`: chiave completa del
          frammento (uso in `certifica_frammento`);
        - `chiave(arr[, dx])`: chiave del singolo campo (comoda per test
          e cache parziali).

        Accetta anche keyword `Fo=, Fx=, Fy=, essenza=, dx=`. Ogni campo
        assente contribuisce come token `None`. `dx` entra nell'hash come
        stringa, cosi' la chiave include N/shape/dx e l'invalidazione e'
        implicita: valori, forma o passo diversi -> chiave diversa (miss).
        L'arrotondamento a 1e-6 stabilizza il rumore numerico.
        """
        Fo = kwargs.get("Fo", None)
        Fx = kwargs.get("Fx", None)
        Fy = kwargs.get("Fy", None)
        essenza = kwargs.get("essenza", None)
        if "dx" in kwargs and dx is None:
            dx = kwargs["dx"]

        def _e_dx(x) -> bool:
            # scalare o mini-sequenza numerica (<=3 valori): passo, non campo
            if x is None or isinstance(x, str):
                return False
            try:
                a = np.asarray(x, dtype=float)
            except (TypeError, ValueError):
                return False
            return bool(a.size <= 3)

        restanti = list(args)
        if restanti and Fo is None and "Fo" not in kwargs:
            Fo = restanti.pop(0)
        if restanti and Fx is None and "Fx" not in kwargs:
            cand = restanti[0]
            if Fy is None and essenza is None and len(restanti) == 1 and _e_dx(cand):
                dx = cand
                restanti.pop(0)
            else:
                Fx = restanti.pop(0)
        if restanti and Fy is None and "Fy" not in kwargs:
            Fy = restanti.pop(0)
        if restanti and essenza is None and "essenza" not in kwargs:
            essenza = restanti.pop(0)
        if restanti and dx is None:
            # quinto posizionale eventuale: dx
            dx = restanti.pop(0)

        h = hashlib.sha256()

        def _aggiorna(a, nome: str) -> None:
            h.update((":" + nome).encode("utf-8"))
            if a is None:
                h.update(b"None")
                return
            arr = np.asarray(a, dtype=float)
            h.update(str(arr.shape).encode("utf-8"))
            h.update(_arrotonda(arr).tobytes())

        _aggiorna(Fo, "Fo")
        _aggiorna(Fx, "Fx")
        _aggiorna(Fy, "Fy")
        _aggiorna(essenza, "essenza")
        h.update(b":dx")
        h.update(str(dx).encode("utf-8"))
        return h.hexdigest()

    def salva(self, chiave: str, payload: dict) -> None:
        """Memorizza (copia di) `payload` sotto `chiave` certificata."""
        self._cache[str(chiave)] = dict(payload)
        self._salvataggi += 1

    def richiama(self, chiave: str) -> tuple[bool, dict | None]:
        """Richiama senza ricalcolare: (hit, payload) o (False, None)."""
        k = str(chiave)
        if k in self._cache:
            self._hits += 1
            return True, self._cache[k]
        self._misses += 1
        return False, None

    def stats(self) -> dict:
        """Conteggi cache: hits, misses, salvataggi (+ elementi)."""
        return {
            "hits": int(self._hits),
            "misses": int(self._misses),
            "salvataggi": int(self._salvataggi),
            "elementi": int(len(self._cache)),
        }


# ---------------------------------------------------------------------------
# 5. Diagnostica monoriga
# ---------------------------------------------------------------------------
def diagnostica_gf(
    quiete: dict | None = None, certificazione: dict | None = None
) -> str:
    """Riga leggibile di diagnostica GF (quiete e/o certificazione).

    Accetta il dict di `verifica_quiete` come primo argomento e/o il dict
    di `certifica_frammento` come secondo (se il primo dict contiene gia'
    `certificato` e il secondo e' None, e' trattato come certificazione).
    Con entrambi None ritorna comunque una riga di stato.
    """
    cert = certificazione
    q = quiete
    if cert is None and isinstance(q, dict) and "certificato" in q:
        cert, q = q, (q.get("quiete") if isinstance(q.get("quiete"), dict) else None)

    parti: list[str] = []
    if isinstance(q, dict) and "attivo" in q:
        parti.append(
            f"quiete={'SI' if q.get('attivo') else 'NO'} "
            f"(err_fx_fo={float(q.get('err_fx_fo', float('nan'))):.4f}, "
            f"err_fo_ess={float(q.get('err_fo_ess', float('nan'))):.4f}, "
            f"nov_rel={float(q.get('novita_rel', float('nan'))):.4f})"
        )
    if isinstance(cert, dict) and "certificato" in cert:
        adatt = cert.get("adattamento")
        adatt_s = "n/d" if adatt is None else f"{float(adatt):.3f}"
        parti.append(
            f"cert={'SI' if cert.get('certificato') else 'NO'} "
            f"(Q={float(cert.get('qualita', float('nan'))):.3f}, "
            f"ad={adatt_s}, nov_rel={float(cert.get('novita_rel', float('nan'))):.4f})"
        )
        if cert.get("motivo"):
            parti.append(str(cert["motivo"]))
    elif isinstance(q, dict) and q.get("motivo"):
        parti.append(str(q["motivo"]))
    if not parti:
        return "GF: nessun esito (quiete/certificazione assenti)"
    return "GF " + " | ".join(parti)
