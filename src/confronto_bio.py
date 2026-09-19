"""Frammento del veloce - confronto bio (LFP/theta-gamma come analogie).

Verifica INTERNA del sistema sintetico + ponte onesto verso dati reali.

Premessa onesta (vedi README §7): questo modulo NON valida nulla di
biologico. `lfp_sintetico` impone per costruzione theta 6 Hz + gamma 45 Hz
modulata dalla novita': i picchi in fig07 sono tautologici. Qui si verifica
solo la coerenza interna (picchi dove imposti, gamma che cresce con la
novita', accoppiamento fase-ampiezza superiore al surrogato) e si fornisce
una funzione di confronto spettrale con un tracciato reale (CSV/EEG) che
restituisce similarita' senza mai dichiarare "validazione".

Dipendenze: numpy + scipy (gia' in requirements.txt).
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import hilbert
    _HA_SCIPY = True
except Exception:  # pragma: no cover
    _HA_SCIPY = False

__all__ = [
    "spettro_potenza",
    "potenza_banda",
    "potenza_relativa_theta_gamma",
    "filtro_banda",
    "indice_pac_theta_gamma",
    "pac_vs_surrogato",
    "similarita_spettrale",
    "valida_sistema_sintetico",
    "confronta_sintetico_vs_reale",
    "carica_eeg_csv",
    "trend_carico",
    "p_permutazione_trend",
]

EPS = 1e-12


# ---------------------------------------------------------------------------
# Spettro
# ---------------------------------------------------------------------------
def spettro_potenza(lfp: np.ndarray, fs: float):
    """Periodogramma semplice via rFFT con finestra di Hann."""
    y = np.asarray(lfp, dtype=float).ravel()
    w = np.hanning(y.size)
    Y = np.fft.rfft(y * w)
    f = np.fft.rfftfreq(y.size, 1.0 / float(fs))
    psd = (np.abs(Y) ** 2) / (np.sum(w ** 2) + EPS)
    return f, psd


def potenza_banda(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    f = np.asarray(freqs, dtype=float)
    p = np.asarray(psd, dtype=float)
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        return 0.0
    return float(np.trapezoid(p[m], f[m]))


def potenza_relativa_theta_gamma(lfp: np.ndarray, fs: float,
                                 banda_theta=(4.0, 8.0),
                                 banda_gamma=(30.0, 60.0)) -> dict:
    """Potenze theta/gamma assolute e relative + rapporto gamma/theta."""
    f, psd = spettro_potenza(lfp, fs)
    p_theta = potenza_banda(f, psd, *banda_theta)
    p_gamma = potenza_banda(f, psd, *banda_gamma)
    p_tot = float(np.trapezoid(psd, f)) + EPS
    return {
        "p_theta": float(p_theta),
        "p_gamma": float(p_gamma),
        "p_tot": float(p_tot),
        "rel_theta": float(p_theta / p_tot),
        "rel_gamma": float(p_gamma / p_tot),
        "rapporto_gamma_theta": float(p_gamma / (p_theta + EPS)),
        "freqs": f,
        "psd": psd,
    }


# ---------------------------------------------------------------------------
# Filtro di banda + PAC (Tort MI)
# ---------------------------------------------------------------------------
def filtro_banda(y: np.ndarray, fs: float, fmin: float, fmax: float) -> np.ndarray:
    """Passabanda ideale via maschera FFT (sufficiente per test sintetici)."""
    y = np.asarray(y, dtype=float).ravel()
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(y.size, 1.0 / float(fs))
    mask = (f >= fmin) & (f <= fmax)
    Yf = np.where(mask, Y, 0.0)
    return np.fft.irfft(Yf, n=y.size)


def _fase_ampiezza(y_theta: np.ndarray, y_gamma: np.ndarray):
    if _HA_SCIPY:
        fase = np.angle(hilbert(y_theta))
        ampl = np.abs(hilbert(y_gamma))
    else:  # fallback senza scipy: quadratura via FFT
        def _analytic(x):
            X = np.fft.fft(x)
            h = np.zeros_like(X)
            n = x.size
            h[0] = 1.0
            if n % 2 == 0:
                h[1:n // 2] = 2.0
                h[n // 2] = 1.0
            else:
                h[1:(n + 1) // 2] = 2.0
            return np.fft.ifft(X * h)
        fase = np.angle(_analytic(y_theta))
        ampl = np.abs(_analytic(y_gamma))
    return fase, ampl


def indice_pac_theta_gamma(lfp: np.ndarray, fs: float,
                           banda_theta=(4.0, 8.0),
                           banda_gamma=(30.0, 60.0),
                           n_bins: int = 18) -> dict:
    """Modulation Index di Tort: KL(distribuzione ampiezza/fase || uniforme).

    Ritorna MI in [0, ~log(n_bins)] normalizzato in [0,1] dividendo per
    log(n_bins), piu' fase, ampiezza e distribuzione per diagnostica.
    """
    y = np.asarray(lfp, dtype=float).ravel()
    yt = filtro_banda(y, fs, *banda_theta)
    yg = filtro_banda(y, fs, *banda_gamma)
    fase, ampl = _fase_ampiezza(yt, yg)
    bins = np.linspace(-np.pi, np.pi, int(n_bins) + 1)
    idx = np.clip(np.digitize(fase, bins) - 1, 0, int(n_bins) - 1)
    prof = np.array([ampl[idx == b].mean() if np.any(idx == b) else 0.0
                     for b in range(int(n_bins))])
    prof = prof / (prof.sum() + EPS)
    uniforme = np.full_like(prof, 1.0 / len(prof))
    mask = prof > 0
    kl = float(np.sum(prof[mask] * np.log(prof[mask] / uniforme[mask])))
    mi = float(kl / (np.log(len(prof)) + EPS))
    return {"mi": mi, "kl": kl, "profilo": prof, "fase": fase, "ampiezza": ampl}


def pac_vs_surrogato(lfp: np.ndarray, fs: float, n_surrogati: int = 20,
                     seed: int = 0) -> dict:
    """MI reale vs distribuzione di surrogati (ampiezza mescolata).

    z = (MI_reale - media_surrogati) / std_surrogati. z > 2 = accoppiamento
    oltre il caso (sul sintetico e' atteso per costruzione: gamma gated da
    theta in `lfp_sintetico`).
    """
    rng = np.random.default_rng(int(seed))
    mi_reale = float(indice_pac_theta_gamma(lfp, fs)["mi"])
    y = np.asarray(lfp, dtype=float).ravel()
    yt = filtro_banda(y, fs, 4.0, 8.0)
    yg = filtro_banda(y, fs, 30.0, 60.0)
    fase, ampl = _fase_ampiezza(yt, yg)
    n_bins = 18
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(fase, bins) - 1, 0, n_bins - 1)
    mis = []
    for _ in range(int(n_surrogati)):
        a_sh = rng.permutation(ampl)
        prof = np.array([a_sh[idx == b].mean() if np.any(idx == b) else 0.0
                         for b in range(n_bins)])
        prof = prof / (prof.sum() + EPS)
        uni = np.full_like(prof, 1.0 / len(prof))
        m = prof > 0
        kl = float(np.sum(prof[m] * np.log(prof[m] / uni[m])))
        mis.append(kl / (np.log(len(prof)) + EPS))
    mis = np.asarray(mis, dtype=float)
    mu, sd = float(mis.mean()), float(mis.std() + EPS)
    return {"mi_reale": mi_reale, "mi_surrogati_media": mu,
            "mi_surrogati_std": sd, "z": float((mi_reale - mu) / sd),
            "mi_surrogati": mis}


# ---------------------------------------------------------------------------
# Similarita' spettrale
# ---------------------------------------------------------------------------
def similarita_spettrale(psd1: np.ndarray, psd2: np.ndarray) -> float:
    """Coseno tra spettri normalizzati in [0,1]."""
    a = np.asarray(psd1, dtype=float).ravel()
    b = np.asarray(psd2, dtype=float).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


# ---------------------------------------------------------------------------
# Validazione INTERNA del sistema sintetico
# ---------------------------------------------------------------------------
def valida_sistema_sintetico(lfp_bassa_novita: np.ndarray,
                             lfp_alta_novita: np.ndarray,
                             fs: float) -> dict:
    """Tre check di coerenza interna (non biologia):

    1. picchi spettrali alle bande imposte (theta 4-8, gamma 30-60);
    2. potenza gamma che cresce con la novita';
    3. PAC reale > surrogati (z > 2).
    """
    checks = {}
    for nome, sig in (("bassa", lfp_bassa_novita), ("alta", lfp_alta_novita)):
        f, psd = spettro_potenza(sig, fs)
        p_th = potenza_banda(f, psd, 4.0, 8.0)
        p_gm = potenza_banda(f, psd, 30.0, 60.0)
        p_tot = float(np.trapezoid(psd, f)) + EPS
        # picco: massimo locale vicino a 6 e 45 Hz
        i6 = int(np.argmin(np.abs(f - 6.0)))
        i45 = int(np.argmin(np.abs(f - 45.0)))
        picco6 = bool(psd[i6] >= np.median(psd[(f >= 2) & (f <= 12)]))
        picco45 = bool(psd[i45] >= np.median(psd[(f >= 20) & (f <= 70)]))
        checks[nome] = {"p_theta": p_th, "p_gamma": p_gm,
                        "rel_gamma": float(p_gm / p_tot),
                        "picco_theta_6Hz": picco6, "picco_gamma_45Hz": picco45}
    gamma_sale = bool(checks["alta"]["p_gamma"] > checks["bassa"]["p_gamma"])
    pac = pac_vs_surrogato(lfp_alta_novita, fs, n_surrogati=20, seed=0)
    ok = bool(checks["alta"]["picco_theta_6Hz"] and checks["alta"]["picco_gamma_45Hz"]
              and gamma_sale and pac["z"] > 2.0)
    motivo = (f"picchi 6/45Hz={checks['alta']['picco_theta_6Hz']}/{checks['alta']['picco_gamma_45Hz']}, "
              f"gamma sale con novita'={gamma_sale} "
              f"({checks['bassa']['p_gamma']:.3g}->{checks['alta']['p_gamma']:.3g}), "
              f"PAC z={pac['z']:.2f} (MI={pac['mi_reale']:.4f})")
    if not ok:
        motivo = "NON coerente (interno): " + motivo
    else:
        motivo = "coerente (interno, NON biologico): " + motivo
    return {"ok_interno": ok, "motivo": motivo, "dettagli": checks, "pac": pac,
            "validazione_biologica": False}


# ---------------------------------------------------------------------------
# Confronto onesto sintetico vs reale
# ---------------------------------------------------------------------------
def confronta_sintetico_vs_reale(lfp_sim: np.ndarray, lfp_reale: np.ndarray,
                                 fs: float) -> dict:
    """Confronto spettrale sim vs tracciato reale (stessa fs).

    Restituisce similarita' coseno + potenze relative. NON dichiara mai
    validazione: il flag `validazione_biologica` e' sempre False; una
    similarita' alta e' compatibilita' spettrale, non prova.
    """
    a = np.asarray(lfp_sim, dtype=float).ravel()
    b = np.asarray(lfp_reale, dtype=float).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    fa, pa = spettro_potenza(a, fs)
    fb, pb = spettro_potenza(b, fs)
    sim = similarita_spettrale(pa, pb)
    ra = potenza_relativa_theta_gamma(a, fs)
    rb = potenza_relativa_theta_gamma(b, fs)
    return {
        "similarita_coseno": float(sim),
        "rel_gamma_sim": float(ra["rel_gamma"]),
        "rel_gamma_reale": float(rb["rel_gamma"]),
        "rel_theta_sim": float(ra["rel_theta"]),
        "rel_theta_reale": float(rb["rel_theta"]),
        "rapporto_gamma_theta_sim": float(ra["rapporto_gamma_theta"]),
        "rapporto_gamma_theta_reale": float(rb["rapporto_gamma_theta"]),
        "motivo": (f"similarita' spettrale coseno={sim:.3f}; "
                   f"gamma/theta sim={ra['rapporto_gamma_theta']:.3f} vs "
                   f"reale={rb['rapporto_gamma_theta']:.3f}. "
                   f"Compatibilita' spettrale, NON validazione biologica."),
        "validazione_biologica": False,
    }


def carica_eeg_csv(path: str, colonna: int = 0, fs: float = 256.0,
                   skip_header: int = 1, delimiter: str = ",") -> dict:
    """Carica un tracciato reale da CSV (una colonna = un canale).

    Formato atteso: righe temporali, colonne canali. Ritorna dict con
    `segnale`, `fs`, `n_campioni`. Solleva FileNotFoundError se assente.
    Pensata per dataset aperti esportati in CSV (OpenNeuro/PhysioNet/TUH
    pre-esportati); nessun download automatico.
    """
    import os
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file EEG non trovato: {path}")
    dati = np.loadtxt(path, delimiter=delimiter, skiprows=skip_header)
    if dati.ndim == 1:
        segnale = dati
    else:
        segnale = dati[:, int(colonna)]
    segnale = np.asarray(segnale, dtype=float).ravel()
    return {"segnale": segnale, "fs": float(fs),
            "n_campioni": int(segnale.size), "path": str(path)}


# ---------------------------------------------------------------------------
# Trend di gruppo sui carichi (Fase 12)
# ---------------------------------------------------------------------------
def trend_carico(matrice: np.ndarray, carichi=(3, 6, 9, 12, 15)) -> dict:
    """Spearman pooled carico vs metrica su matrice soggetti×carichi.

    `matrice[i, j]` = mediana del soggetto i al carico j. Ritorna rho, p
    (asintotico) e medie di gruppo per carico. Test esplorativo di gruppo,
    non validazione.
    """
    from scipy.stats import spearmanr
    M = np.asarray(matrice, dtype=float)
    car = np.asarray(list(carichi), dtype=float)
    x = np.repeat(car, M.shape[0])
    y = np.concatenate([M[:, j] for j in range(M.shape[1])])
    mask = np.isfinite(y)
    rho, p = spearmanr(x[mask], y[mask])
    return {"rho": float(rho), "p": float(p),
            "medie_gruppo": [float(np.nanmean(M[:, j])) for j in range(M.shape[1])],
            "n_soggetti": int(M.shape[0])}


def p_permutazione_trend(matrice: np.ndarray, carichi=(3, 6, 9, 12, 15),
                         n_perm: int = 2000, seed: int = 0) -> dict:
    """p-value per permutazione: mischia le etichette carico entro soggetto.

    Statistica = |rho| di Spearman pooled (two-sided). Ritorna p con
    correzione +1 (conservativa) e rho osservato.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(int(seed))
    M = np.asarray(matrice, dtype=float)
    car = np.asarray(list(carichi), dtype=float)
    x = np.repeat(car, M.shape[0])
    y0 = np.concatenate([M[:, j] for j in range(M.shape[1])])
    rho0 = abs(float(spearmanr(x[np.isfinite(y0)], y0[np.isfinite(y0)])[0]))
    cnt = 0
    for _ in range(int(n_perm)):
        Mp = np.array([rng.permutation(row) for row in M])
        yp = np.concatenate([Mp[:, j] for j in range(Mp.shape[1])])
        m = np.isfinite(yp)
        if abs(float(spearmanr(x[m], yp[m])[0])) >= rho0:
            cnt += 1
    return {"rho_osservato": float(rho0),
            "p_perm": float((cnt + 1) / (int(n_perm) + 1)),
            "n_perm": int(n_perm)}
