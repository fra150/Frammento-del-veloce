"""Test degli studi di robustezza (src/studi.py). Veloci (N piccolo)."""

import os

from src.frammento_2d import Param, simula
from src.studi import (
    COLONNE_SWEEP,
    config_ablazione,
    config_sweep,
    esegui,
    safe_corr,
    salva_csv,
    tabella_md,
    tempo_recupero,
    valuta,
)


def _mini_param(**kw):
    base = dict(N=16, D0=0.05, Dx=0.01, Dy=0.002, alpha=3.0, kappa_x=0.6,
                beta=0.9, K=1.2, gamma=0.02, kappa_y=0.25, seed=7)
    base.update(kw)
    return Param(**base)


def test_safe_corr():
    import numpy as np
    assert safe_corr([1.0, 2.0], [1.0, 2.0]) == 1.0
    assert safe_corr([1.0, 1.0], [2.0, 3.0]) == 0.0
    assert -1.0 <= safe_corr([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) <= 1.0


def test_stato_iniziale_rispettato():
    import numpy as np
    p = _mini_param()
    zeri = np.zeros((p.N, p.N))
    # T < dt -> nessuno step: lo snapshot coincide con lo stato iniziale
    snap = simula(p, T=0.001, salva_ogni=1, stato_iniziale={"Fx": zeri})
    assert abs(snap["Fx"][0].max()) == 0.0


def test_valuta_chiavi_e_range():
    p = _mini_param()
    r = valuta(p, T=0.02, salva_ogni=1, T_rec=0.02)
    for chiave in COLONNE_SWEEP:
        if chiave == "config":  # la aggiunge esegui(), non valuta()
            continue
        assert chiave in r, chiave
    assert abs(r["massa_g0_fin"] - 1.0) < 1e-9
    assert r["qualita_fin"] <= 1.0
    assert -1.0 <= r["fedelta"] <= 1.0
    assert r["fedelta"] < 1.0  # con rumore, gy stocastica != deterministica
    assert 0.0 <= r["fraz_dVdt_nonpos"] <= 1.0
    assert isinstance(r["recuperato"], bool)


def test_fedelta_unitaria_senza_rumore():
    p = _mini_param(gamma=0.0)
    r = valuta(p, T=0.02, salva_ogni=1, T_rec=0.02)
    assert r["fedelta"] == 1.0


def test_tempo_recupero():
    p = _mini_param()
    base = simula(p, T=0.02, salva_ogni=1)
    t_rec, ok = tempo_recupero(p, base, T_rec=0.02, salva_ogni=1)
    assert 0.0 <= t_rec <= 0.02
    assert isinstance(ok, bool)


def test_sweep_mini_scrive_csv(tmp_path):
    cfgs = [
        {"nome": "mini-A", "param": {**_mini_param().__dict__}, "sim": {}},
        {"nome": "mini-B", "param": {**_mini_param(gamma=0.0).__dict__},
         "sim": {"protocollo": "rilassamento"}},
    ]
    righe = esegui(cfgs, T=0.02, salva_ogni=1, seed=7)
    assert len(righe) == 2
    path = os.path.join(str(tmp_path), "sweep.csv")
    salva_csv(righe, COLONNE_SWEEP, path)
    assert os.path.isfile(path)
    md = tabella_md(righe, COLONNE_SWEEP)
    assert "mini-A" in md and "mini-B" in md


def test_configurazioni_complete():
    assert len(config_sweep()) == 7
    assert len(config_ablazione()) == 6
    for cfg in config_sweep() + config_ablazione():
        Param(**cfg["param"])  # parametri validi
        assert cfg["nome"]
