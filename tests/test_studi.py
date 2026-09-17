"""Test degli studi di robustezza (src/studi.py). Veloci (N piccolo)."""

import os

import numpy as np

from src.frammento_2d import Param, simula, verifica_invarianza
from src.studi import (
    COLONNE_SWEEP,
    SEEDS_DEFAULT,
    _adattamento,
    config_ablazione,
    config_sweep,
    esegui,
    safe_corr,
    salva_csv,
    tabella_md,
    tempo_recupero,
    valuta,
    valuta_multiseed,
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


def test_lambda_g_costante_tra_livelli():
    # invarianza geometrica: lambda_g quasi costante su 25x di D
    # (serve finestra lunga: campionamento ogni 40 step)
    p = _mini_param(N=64)
    ver = verifica_invarianza(p, T=0.06)
    assert ver["cv"] < 0.15, ver
    assert 0.5 < ver["media"] < 2.5, ver


def test_adattamento_nondegenere_cresce_con_alpha():
    p0 = _mini_param(alpha=0.0, kappa_x=0.0)
    p1 = _mini_param(alpha=3.0)
    p2 = _mini_param(alpha=8.0)
    s0 = simula(p0, T=0.03, salva_ogni=2)
    s1 = simula(p1, T=0.03, salva_ogni=2)
    s2 = simula(p2, T=0.03, salva_ogni=2)
    a0 = _adattamento(p0, s0)
    a1 = _adattamento(p1, s1)
    a2 = _adattamento(p2, s2)
    assert not np.allclose([a0, a1, a2], -0.04, atol=0.01)
    assert a0 < a1 < a2, (a0, a1, a2)


def test_multiseed_media_std_due_seed():
    p = _mini_param()
    r = valuta_multiseed(p, seeds=(7, 11), T=0.02, salva_ogni=1)
    assert r["n_seed"] == 2
    assert 0.0 <= r["fedelta_mean"] <= 1.0
    assert r["fedelta_std"] >= 0.0
    assert 0.0 <= r["recuperato_frac"] <= 1.0


def test_main_multiseed_mini_scrive_file(tmp_path):
    from src.studi import main_ablazione_multiseed, main_sweep_multiseed
    out = str(tmp_path)
    rs = main_sweep_multiseed(N=16, T=0.2, out_dir=out, seeds=(7, 11))
    ra = main_ablazione_multiseed(N=16, T=0.2, out_dir=out, seeds=(7, 11))
    assert len(rs) == 7 and len(ra) == 6
    for nome in ("studio_parametri_multiseed.csv",
                 "studio_parametri_multiseed.md",
                 "ablazione_multiseed.csv", "ablazione_multiseed.md",
                 "fig08_ablazione.png"):
        assert os.path.isfile(os.path.join(out, nome)), nome
