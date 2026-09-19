"""Test di coerenza interna LFP/theta-gamma + ponte verso dato reale.

NON sono validazione biologica: verificano che il sistema sintetico faccia
quello che dichiara (picchi imposti, gamma modulata da novita', PAC oltre il
caso) e che il confronto spettrale sim-vs-reale sia strutturato e onesto
(flag validazione_biologica sempre False).

Veloci (<5 s), nessuna figura, nessun download.
"""

import numpy as np

from src.confronto_bio import (
    confronta_sintetico_vs_reale,
    indice_pac_theta_gamma,
    pac_vs_surrogato,
    potenza_relativa_theta_gamma,
    similarita_spettrale,
    spettro_potenza,
    valida_sistema_sintetico,
)
from src.frammento_2d import Param, essenza, lfp_sintetico


def _fx_bassa_alta_novita():
    p = Param(N=16, seed=7)
    ess = essenza(p)
    # bassa novita': campo quasi piatto -> std piccola
    fx_bassa = np.full_like(ess, float(ess.mean()))
    # alta novita': essenza + bump localizzato -> std grande
    fx_alta = ess.copy()
    c = p.N // 2
    fx_alta[c - 2:c + 2, c - 2:c + 2] += 2.0
    return p, fx_bassa, fx_alta


def test_picchi_spettrali_dove_imposti():
    p = Param(N=16, seed=7)
    ess = essenza(p)
    sig = lfp_sintetico(p, ess, fs=1000.0, durata=1.0, seed=3)
    f, psd = spettro_potenza(sig["lfp"], sig["fs"])
    i6 = int(np.argmin(np.abs(f - 6.0)))
    i45 = int(np.argmin(np.abs(f - 45.0)))
    # il picco imposto deve emergere sopra la mediana della banda vicina
    assert psd[i6] >= np.median(psd[(f >= 2) & (f <= 12)])
    assert psd[i45] >= np.median(psd[(f >= 20) & (f <= 70)])


def test_gamma_cresce_con_novita():
    p, fx_bassa, fx_alta = _fx_bassa_alta_novita()
    s_b = lfp_sintetico(p, fx_bassa, fs=1000.0, durata=1.0, seed=3)
    s_a = lfp_sintetico(p, fx_alta, fs=1000.0, durata=1.0, seed=3)
    assert s_a["novita"] > s_b["novita"]
    rb = potenza_relativa_theta_gamma(s_b["lfp"], 1000.0)
    ra = potenza_relativa_theta_gamma(s_a["lfp"], 1000.0)
    assert ra["p_gamma"] > rb["p_gamma"]


def test_pac_reale_maggiore_surrogato():
    p = Param(N=16, seed=7)
    ess = essenza(p)
    sig = lfp_sintetico(p, ess, fs=1000.0, durata=2.0, seed=3)
    out = pac_vs_surrogato(sig["lfp"], sig["fs"], n_surrogati=20, seed=0)
    assert out["mi_reale"] > out["mi_surrogati_media"]
    assert out["z"] > 2.0


def test_valida_sistema_sintetico_ok():
    p, fx_bassa, fx_alta = _fx_bassa_alta_novita()
    s_b = lfp_sintetico(p, fx_bassa, fs=1000.0, durata=1.0, seed=3)
    s_a = lfp_sintetico(p, fx_alta, fs=1000.0, durata=1.0, seed=3)
    out = valida_sistema_sintetico(s_b["lfp"], s_a["lfp"], fs=1000.0)
    assert out["validazione_biologica"] is False
    assert out["ok_interno"] is True


def test_simile_a_se_stesso_diverso_da_rumore():
    p = Param(N=16, seed=7)
    ess = essenza(p)
    s1 = lfp_sintetico(p, ess, fs=500.0, durata=1.0, seed=3)
    s2 = lfp_sintetico(p, ess, fs=500.0, durata=1.0, seed=4)
    rng = np.random.default_rng(0)
    rumore = rng.standard_normal(s1["lfp"].size)
    _, p1 = spettro_potenza(s1["lfp"], 500.0)
    _, p2 = spettro_potenza(s2["lfp"], 500.0)
    _, pr = spettro_potenza(rumore, 500.0)
    sim_stesso = similarita_spettrale(p1, p2)
    sim_rumore = similarita_spettrale(p1, pr)
    assert 0.0 <= sim_rumore <= 1.0
    assert sim_stesso > sim_rumore


def test_confronto_sintetico_vs_reale_onesto(tmp_path):
    # "reale" qui = rumore rosa stand-in, MAI spacciato per EEG vero
    p = Param(N=16, seed=7)
    ess = essenza(p)
    sig = lfp_sintetico(p, ess, fs=500.0, durata=1.0, seed=3)
    rng = np.random.default_rng(1)
    standin = np.cumsum(rng.standard_normal(sig["lfp"].size))
    standin = standin / (np.std(standin) + 1e-12)
    out = confronta_sintetico_vs_reale(sig["lfp"], standin, fs=500.0)
    assert out["validazione_biologica"] is False
    assert 0.0 <= out["similarita_coseno"] <= 1.0
    assert "NON validazione" in out["motivo"]


def test_carica_eeg_csv(tmp_path):
    import numpy as np
    from src.confronto_bio import carica_eeg_csv
    fp = tmp_path / "eeg_prova.csv"
    rng = np.random.default_rng(0)
    dati = rng.standard_normal((500, 2))
    np.savetxt(fp, dati, delimiter=",", header="ch1,ch2", comments="")
    out = carica_eeg_csv(str(fp), colonna=1, fs=256.0)
    assert out["fs"] == 256.0
    assert out["n_campioni"] == 500
    assert out["segnale"].shape == (500,)


def test_mi_stabile_e_limitato():
    p = Param(N=16, seed=7)
    ess = essenza(p)
    sig = lfp_sintetico(p, ess, fs=500.0, durata=1.0, seed=3)
    mi = indice_pac_theta_gamma(sig["lfp"], sig["fs"])["mi"]
    assert np.isfinite(mi) and 0.0 <= mi <= 1.0
