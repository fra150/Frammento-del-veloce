"""Frammento del veloce - libreria di simulazione numerica.

Modello a tre livelli geometrici per la diffusione dinamica della memoria:
    g0 : essenza perfetta / invariante  (R = 0, solo diffusione conservativa)
    gx : operativita' vincolata agli input (accoppiamento con x(t))
    gy : novita' controllata (reazione non lineare + rumore, vincolata da g0 e gx)

Riferimento concettuale: reazione-diffusione con misura invariante, funzione di
Lyapunov e invarianza n = v (numero di elementi = velocita' di diffusione,
a meno della costante di scala fissata dal livello g0).

Autore della ricerca: dr. Bulla Francesco (Catania, 17/09/2026).
Implementazione: uso didattico/scientifico, dipendenze solo numpy/scipy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------------
# 1. Parametri del modello
# ----------------------------------------------------------------------------
@dataclass
class Param:
    """Coefficienti delle equazioni di diffusione-reazione dei tre livelli."""

    N: int = 96                 # punti di griglia per lato
    L: float = 1.0              # lato del dominio (toro -> varieta' compatta)
    D0: float = 0.05            # diffusione g0  (D_g0 >> D_gx >> D_gy)
    Dx: float = 0.01            # diffusione gx
    Dy: float = 0.002           # diffusione gy
    alpha: float = 3.0          # accoppiamento gx <-> input x
    kappa_x: float = 0.6        # richiamo di gx verso l'essenza g0
    beta: float = 0.9           # crescita creativa (logistica) in gy
    K: float = 1.2              # capacita' massima di novita' compatibile
    gamma: float = 0.02         # intensita' del rumore creativo
    kappa_y: float = 0.25       # smorzamento / vincolo di gy
    soglia: float = 0.02        # soglia di attivazione del gate gx
    seed: int = 7

    @property
    def dx(self) -> float:
        return self.L / self.N

    def dt_stabile(self, fattore: float = 0.4) -> float:
        """Passo temporale massimo per Eulero esplicito: dt <= dx^2/(4 D_max)."""
        return fattore * self.dx ** 2 / (4.0 * max(self.D0, self.Dx, self.Dy))


# ----------------------------------------------------------------------------
# 2. Geometria di base: griglia, laplaciano, essenza g0, input x
# ----------------------------------------------------------------------------
def griglia(p: Param):
    """Coordinate spaziali (X, Y) su dominio periodico [0, L)^2."""
    c = (np.arange(p.N) + 0.5) * p.dx
    X, Y = np.meshgrid(c, c, indexing="ij")
    return X, Y


def laplaciano(F: np.ndarray, dx: float) -> np.ndarray:
    """Laplaciano su griglia periodica (varieta' compatta, misura invariante)."""
    return (
        np.roll(F, 1, 0) + np.roll(F, -1, 0)
        + np.roll(F, 1, 1) + np.roll(F, -1, 1)
        - 4.0 * F
    ) / dx ** 2


def _gaussiana(X, Y, cx, cy, s):
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * s ** 2))


def essenza(p: Param, massa: float = 1.0) -> np.ndarray:
    """Distribuzione ideale del Frammento secondo g0 (due tracce mnestiche).

    Rappresenta la 'geometria g0': nucleo immutabile, 100% qualita'.
    """
    X, Y = griglia(p)
    Z = (_gaussiana(X, Y, 0.32 * p.L, 0.34 * p.L, 0.075 * p.L)
         + 0.75 * _gaussiana(X, Y, 0.70 * p.L, 0.66 * p.L, 0.055 * p.L))
    return massa * Z / (Z.sum() * p.dx ** 2)


def input_field(p: Param, t: float, ampiezza: float = 1.0) -> np.ndarray:
    """Stimolo esterno x(t): traccia mobile che percorre una curva di Lissajous."""
    X, Y = griglia(p)
    cx = p.L * (0.5 + 0.34 * np.sin(2.0 * np.pi * t / 0.30))
    cy = p.L * (0.5 + 0.34 * np.cos(2.0 * np.pi * t / 0.45))
    Z = _gaussiana(X, Y, cx, cy, 0.060 * p.L)
    return ampiezza * Z / (Z.max() + EPS)


# ----------------------------------------------------------------------------
# 3. Integratore: sistema accoppiato a tre livelli (Eulero-Maruyama)
# ----------------------------------------------------------------------------
def simula(
    p: Param,
    T: float = 0.30,
    dt: float | None = None,
    stocastico: bool = True,
    protocollo: str = "stimolo",       # 'stimolo' | 'rilassamento'
    salva_ogni: int = 50,
    rumore_bianco: bool = True,
    stato_iniziale: dict | None = None,  # {'F0':..., 'Fx':..., 'Fy':...}
) -> dict:
    """Integra il sistema accoppiato F0 (g0), Fx (gx), Fy (gy).

    equazioni (SDE in forma di Ito'):
        dF0 = D0 lap(F0) dt                                   (R_g0 = 0)
        dFx = [Dx lap(Fx) + alpha (Fin - Fx) + kx (F0 - Fx)] dt
        dFy = [Dy lap(Fy) + beta Fy (1 - Fy/K) G - ky Fy] dt + gamma G dW
    con G = gate di compatibilita' (gx attivo e supporto di g0 non nullo),
    dW = xi*sqrt(dt), xi = N(0,1) bianco oppure OU a varianza unitaria.
    Schema: Eulero-Maruyama (deterministico = Eulero esplicito).

    Se `stato_iniziale` e' fornito (chiavi 'F0', 'Fx', 'Fy'), la simulazione
    riparte da quello stato invece che dall'essenza: serve per esperimenti
    di perturbazione/recupero (twin experiment).
    """
    dt = dt or p.dt_stabile()
    rng = np.random.default_rng(p.seed)
    dx = p.dx
    sqrt_dt = float(np.sqrt(dt))
    tau_ou = 0.05  # costante di tempo OU (stessa di prima, ora a varianza unitaria)

    F0 = essenza(p).copy()              # essenza g0, 100% qualita'
    Fx = essenza(p).copy()              # operativita' allineata all'ingresso
    mask = (essenza(p) > 1e-3).astype(float)   # vincolo strutturale di g0
    Fy = 0.05 * mask * rng.random((p.N, p.N))  # germe di novita'
    if stato_iniziale:
        if "F0" in stato_iniziale:
            F0 = np.asarray(stato_iniziale["F0"], dtype=float).copy()
        if "Fx" in stato_iniziale:
            Fx = np.asarray(stato_iniziale["Fx"], dtype=float).copy()
        if "Fy" in stato_iniziale:
            Fy = np.asarray(stato_iniziale["Fy"], dtype=float).copy()

    nsteps = int(T / dt)

    # stato OU locale (evita variabile statica su funzione in caso di N diversi)
    # normalizzato a varianza stazionaria unitaria: Var[ou] -> 1
    ou = np.zeros((p.N, p.N))

    snap = {"t": [], "F0": [], "Fx": [], "Fy": []}
    diag = {"t": [], "massa0": [], "massaX": [], "massaY": [],
            "novita": [], "gate_medio": [], "V": []}

    for k in range(nsteps + 1):
        t = k * dt

        if protocollo == "stimolo":
            Fin = input_field(p, t)
        else:
            Fin = np.zeros_like(F0)

        gate = np.clip(Fx / (1.2 * Fx.max() + EPS), 0.0, 1.0) * mask

        R_x = p.alpha * (Fin - Fx) + p.kappa_x * (F0 - Fx)
        R_y_reattivo = p.beta * Fy * (1.0 - Fy / p.K) * gate - p.kappa_y * Fy
        # Rumore Euler-Maruyama: incremento Wiener ~ sqrt(dt), NON O(dt).
        # Prima il termine entrava come dt*gamma*G*xi (O(dt), invisibile);
        # ora entra come sqrt(dt)*gamma*G*xi come da teoria SDE.
        dW_y = None
        if stocastico:
            if rumore_bianco:
                xi = rng.standard_normal((p.N, p.N))
            else:  # OU a varianza unitaria: d(ou) = -ou/tau dt + sqrt(2/tau) dW
                ou += (-ou * dt / tau_ou
                       + np.sqrt(2.0 * dt / tau_ou)
                       * rng.standard_normal((p.N, p.N)))
                xi = ou
            dW_y = sqrt_dt * p.gamma * gate * xi

        if k < nsteps:
            F0 = F0 + dt * p.D0 * laplaciano(F0, dx)
            Fx = Fx + dt * (p.Dx * laplaciano(Fx, dx) + R_x)
            Fy = Fy + dt * (p.Dy * laplaciano(Fy, dx) + R_y_reattivo)
            if dW_y is not None:
                Fy = Fy + dW_y
            # densita' non negativa (come nel 1D): rettifica il rumore,
            # cosi' gamma aumenta davvero la novita' media
            np.maximum(Fy, 0.0, out=Fy)

        if k % salva_ogni == 0:
            snap["t"].append(t)
            snap["F0"].append(F0.copy())
            snap["Fx"].append(Fx.copy())
            snap["Fy"].append(Fy.copy())
            diag["t"].append(t)
            diag["massa0"].append(F0.sum() * dx ** 2)
            diag["massaX"].append(Fx.sum() * dx ** 2)
            diag["massaY"].append(Fy.sum() * dx ** 2)
            diag["novita"].append(float((Fy * mask).sum() * dx ** 2))
            diag["gate_medio"].append(float(gate.mean()))
            diag["V"].append(lyapunov(F0, Fx, Fy, F0))

    for kk in ("t", "massa0", "massaX", "massaY", "novita", "gate_medio", "V"):
        diag[kk] = np.asarray(diag[kk])
    snap["t"] = np.asarray(snap["t"])
    snap["essenza"] = essenza(p)
    snap["param"] = p
    snap["diag"] = diag
    return snap


# ----------------------------------------------------------------------------
# 4. Funzione di Lyapunov e invarianza n = v
# ----------------------------------------------------------------------------
def lyapunov(F0, Fx, Fy, F_essenza, w=(0.5, 0.3, 0.2)):
    """Candidato di Lyapunov: distanza pesata dall'attrattore dei tre livelli.

    V = w0 ||F0 - F0*||^2 + wx ||Fx - F0||^2 + wy ||Fy||^2
    con F0* = equilibrio omogeneo della diffusione pura di g0 (non, in
    generale, un equilibrio dell'intero sistema accoppiato: la verifica
    dV/dt <= 0 e' quindi numerica, sulle traiettorie simulate, e non una
    dimostrazione di stabilita' globale del modello matematico).
    """
    dx = 1.0 / np.sqrt(F0.size)
    F0_star = np.full_like(F0, F0.mean())
    V0 = np.sum((F0 - F0_star) ** 2) * dx ** 2
    Vx = np.sum((Fx - F0) ** 2) * dx ** 2
    Vy = np.sum(Fy ** 2) * dx ** 2
    return w[0] * V0 + w[1] * Vx + w[2] * Vy


def derivata_numerica(t: np.ndarray, V: np.ndarray) -> np.ndarray:
    """dV/dt stimata alle differenze finite (centrate all'interno)."""
    t = np.asarray(t, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.size < 2:
        return np.zeros_like(V)  # un solo snapshot: nessuna pendenza stimabile
    return np.gradient(V, t)


def esperimento_diffusione(p: Param, D: float, T: float = 0.06,
                           dt: float | None = None) -> dict:
    """Sorgente puntiforme su toro: stima D dal momento secondo <r^2> = 4 D t."""
    dt = dt or p.dt_stabile()
    X, Y = griglia(p)
    F = essenza(p, massa=1.0).copy()
    x0, y0 = 0.32 * p.L, 0.34 * p.L
    ts, r2 = [], []
    nsteps = int(T / dt)
    for k in range(nsteps + 1):
        if k % 40 == 0:
            dX = (X - x0 + p.L / 2) % p.L - p.L / 2
            dY = (Y - y0 + p.L / 2) % p.L - p.L / 2
            massa = F.sum() * p.dx ** 2
            ts.append(k * dt)
            r2.append(float((F * (dX ** 2 + dY ** 2)).sum() * p.dx ** 2 / massa))
        if k < nsteps:
            F = F + dt * D * laplaciano(F, p.dx)
    ts, r2 = np.asarray(ts), np.asarray(r2)
    A = np.vstack([ts, np.ones_like(ts)]).T
    coef, *_ = np.linalg.lstsq(A, r2, rcond=None)
    D_stimato = coef[0] / 4.0
    return {"t": ts, "r2": r2, "D_vero": D, "D_stimato": D_stimato,
            "r2_medio_t": float((r2[-1] - r2[0]) / max(ts[-1] - ts[0], EPS))}


def invariante_nv(p: Param, D: float, T: float = 0.06) -> dict:
    """Verifica numerica dell'invarianza in forma normalizzata.

    n  = massa totale del Frammento (numero di elementi)
    v  = velocita' di diffusione misurata da ``<r^2> = 4 D t``
    v_tilde = v / D_g0 : velocita' adimensionale nella scala fissata da g0.

    La forma 'n = v' e' un principio qualitativo di bilanciamento (n e v hanno
    dimensioni diverse). La grandezza indipendente dalla geometria e':

        lambda_g = D_vero / v   (efficienza numerica inversa)
        eta_g    = v / D_vero = 1/lambda_g

    Se lo schema numerico preserva lo scaling diffusivo, lambda_g e' costante
    al variare di D (livelli g0/gx/gy): v_tilde scala linearmente con D/D0.
    La vecchia forma n = lambda*v_tilde con lambda=1/v_tilde per livello e'
    tautologica e NON va usata come prova di invarianza.
    """
    e = esperimento_diffusione(p, D, T)
    n = 1.0                       # massa iniziale normalizzata del Frammento
    v = e["r2_medio_t"] / 4.0     # coefficiente di diffusione efficace misurato
    v_tilde = v / p.D0            # scala adimensionale fissata da g0
    lambda_g = float(D / max(v, EPS))
    eta_g = float(v / max(D, EPS))
    return {"n": n, "v": v, "v_tilde": v_tilde, "D_vero": D,
            "D_stimato": e["D_stimato"], "lambda_g": lambda_g, "eta_g": eta_g}


def verifica_invarianza(p: Param, T: float = 0.06) -> dict:
    """Costanza di lambda_g sui tre livelli (prova di invarianza geometrica).

    Ritorna lambda per g0/gx/gy, media, std e CV. Invarianza = CV piccolo
    a fronte di D che varia di >10x.
    """
    Ds = {"g0": p.D0, "gx": p.Dx, "gy": p.Dy}
    lam = {}
    for nome, D in Ds.items():
        lam[nome] = invariante_nv(p, D, T)["lambda_g"]
    vals = np.array([lam["g0"], lam["gx"], lam["gy"]], dtype=float)
    media = float(vals.mean())
    std = float(vals.std(ddof=0))
    return {"lambda": lam, "media": media, "std": std,
            "cv": float(std / max(media, EPS)),
            "d_min": min(Ds.values()), "d_max": max(Ds.values())}


# ----------------------------------------------------------------------------
# 5. Novita' controllata: pattern di Turing in gy vincolati da g0 e gx
# ----------------------------------------------------------------------------
def turing_gy(p: Param, passi: int = 12000, dt: float = 2.5e-4,
              Da: float = 0.005, Dh: float = 0.20,
              rho: float = 0.02, mu_a: float = 0.02, mu_h: float = 0.05):
    """Modello attivatore-inibitore (Gierer-Meinhardt) nel livello gy.

    Il pattern (novita' controllata) nasce solo dentro il supporto di g0 e con
    intensita' modulata dal gate di gx -> nessuna deriva caotica, novita' ma
    vincolata alla struttura del ricordo.
    """
    rng = np.random.default_rng(p.seed)
    mask = (essenza(p) > 1e-3).astype(float)
    a = 0.5 * mask + 0.02 * rng.random((p.N, p.N))
    h = 0.5 * mask + 0.02 * rng.random((p.N, p.N))
    for k in range(passi):
        ra = rho * a ** 2 / (h + EPS) - mu_a * a
        rh = rho * a ** 2 - mu_h * h
        a = a + dt * (Da * laplaciano(a, p.dx) + ra)
        h = h + dt * (Dh * laplaciano(h, p.dx) + rh)
        a = np.clip(a, 0.0, 50.0)
        h = np.clip(h, 0.0, 50.0)
        if k % 2000 == 0:
            a *= mask
    a = a * mask
    return {"a": a, "h": h, "mask": mask}


# ----------------------------------------------------------------------------
# 6. Metriche di qualita', continuita' e fedelta' del ricordo
# ----------------------------------------------------------------------------
def metriche(snap: dict) -> dict:
    """Qualita' (somiglianza a g0), continuita' (coerenza temporale), fedelta'."""
    ess = snap["essenza"]
    dx2 = (1.0 / snap["param"].N) ** 2
    q = []
    for Fx in snap["Fx"]:
        q.append(1.0 - np.linalg.norm(Fx - ess) / (np.linalg.norm(ess) + EPS))
    Fx = np.asarray(snap["Fx"])
    c = []
    for i in range(1, len(Fx)):
        num = float(np.sum(Fx[i] * Fx[i - 1]))
        den = float(np.linalg.norm(Fx[i]) * np.linalg.norm(Fx[i - 1])) + EPS
        c.append(num / den)
    return {"t": snap["t"][1:], "qualita": np.asarray(q[1:]),
            "continuita": np.asarray(c), "dx2": dx2}


def fedelta(snap_a: dict, snap_b: dict) -> float:
    """Fedelta': correlazione tra ricordo senza rumore e ricordo con rumore."""
    A = np.asarray(snap_a["Fx"][-1]).ravel()
    B = np.asarray(snap_b["Fx"][-1]).ravel()
    return float(np.corrcoef(A, B)[0, 1])


# ----------------------------------------------------------------------------
# 7. Protocollo neurofisiologico simulato: LFP con ritmi theta/gamma
# ----------------------------------------------------------------------------
def lfp_sintetico(p: Param, Fx: np.ndarray, fs: float = 1000.0, durata: float = 2.0,
                  seed: int = 3):
    """SEEG/LFP sintetico: contenuto theta (memoria di lavoro) + gamma (binding).

    L'ampiezza del ritmo gamma e' modulata dalla 'novita' controllata' locale,
    come previsto dal modello spettrale della memoria (continuum latente-attivo).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, durata, 1.0 / fs)
    theta = np.sin(2 * np.pi * 6.0 * t)
    novita = float(np.clip(Fx.std() / (Fx.mean() + EPS), 0, 1))
    gamma = (0.3 + 0.7 * novita) * np.sin(2 * np.pi * 45.0 * t) * (0.5 + 0.5 * theta)
    lfp = theta + gamma + 0.15 * rng.standard_normal(t.size)
    return {"t": t, "lfp": lfp, "fs": fs, "novita": novita}


# ----------------------------------------------------------------------------
# 8. Riepilogo diagnostico
# ----------------------------------------------------------------------------
def riepilogo(snap: dict) -> str:
    d = snap["diag"]
    dV = derivata_numerica(d["t"], d["V"])
    righe = [
        "--- FRAMMENTO DEL VELOCE : diagnostica numerica ---",
        f"griglia            : {snap['param'].N} x {snap['param'].N}, "
        f"dt = {snap['param'].dt_stabile():.2e}",
        f"massa g0  iniziale : {d['massa0'][0]:.6f} | finale: {d['massa0'][-1]:.6f}",
        f"massa gx  iniziale : {d['massaX'][0]:.6f} | finale: {d['massaX'][-1]:.6f}",
        f"novita' gy (fine)  : {d['novita'][-1]:.6f}",
        f"gate gx medio      : {d['gate_medio'][-1]:.4f}",
        f"Lyapunov V(0)      : {d['V'][0]:.6f} | V(fine): {d['V'][-1]:.6f}",
        f"frazione dV/dt<=0  : {float(np.mean(dV <= 1e-9)) * 100:.1f} %",
    ]
    return "\n".join(righe)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Simulazione Frammento del veloce (2D)")
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--T", type=float, default=0.30)
    ap.add_argument("--protocollo", default="stimolo",
                    choices=["stimolo", "rilassamento"])
    args = ap.parse_args()

    p = Param(N=args.N)
    snap = simula(p, T=args.T, protocollo=args.protocollo)
    print(riepilogo(snap))
    for nome, D in (("g0", p.D0), ("gx", p.Dx), ("gy", p.Dy)):
        inv = invariante_nv(p, D)
        print(f"{nome:>2}: n = {inv['n']:.4f}  v = {inv['v']:.6f} "
              f" v~ = {inv['v_tilde']:.4f}  D_stimato = {inv['D_stimato']:.6f}"
              f" (vero {D:.6f})")
