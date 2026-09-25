"""Fase 17 — test fast degli stress fuori regime (solo misura)."""

import os

import numpy as np
import pytest

from src.fase17 import (
    dt_sicuro,
    valuta_tempi_lunghi,
    salva_report_tempi,
    valuta_griglie,
    salva_report_griglie,
    valuta_diffusione,
    salva_report_diffusione,
    valuta_cue_nonstrutturati,
    salva_report_cue,
    campo_naturale,
    valuta_reali,
    salva_report_reali,
    TIPI_CUE,
    CAMP_REALI,
)
from src.frammento_2d import Param


def test_dt_sicuro_base_uguale_stabile():
    p = Param(N=48, seed=7)
    assert dt_sicuro(p) == pytest.approx(p.dt_stabile())


def test_dt_sicuro_zero_definito():
    p = Param(N=32, seed=7, D0=0.0, Dx=0.0, Dy=0.0)
    dt = dt_sicuro(p)
    assert np.isfinite(dt) and dt > 0


def test_tempi_lunghi_due_orizzonti():
    righe = valuta_tempi_lunghi(N=16, T_list=(0.05, 0.10), seed=7)
    assert len(righe) == 2
    for r in righe:
        assert r["finiti"]
        assert r["fraz_dV_nonpos"] >= 0.9
        assert abs(r["drift_massa_g0"]) < 1e-9


def test_tempi_report_tmp(tmp_path):
    righe = valuta_tempi_lunghi(N=16, T_list=(0.05, 0.10), seed=7)
    out = salva_report_tempi(righe, out_dir=str(tmp_path), N=16, seed=7)
    for k in ("csv", "md", "fig"):
        assert os.path.isfile(out[k])


def test_griglie_piccole_stabili():
    righe = valuta_griglie(N_list=(16, 32), T=0.05, seed=7,
                           sonda_rete=False)
    assert len(righe) == 2
    for r in righe:
        assert r["finiti"]
        assert r["mem_slot_bytes"] == r["N"] * r["N"] * 8 * 4
    assert righe[1]["wall_s"] >= 0.0


def test_griglie_report_tmp(tmp_path):
    righe = valuta_griglie(N_list=(16, 32), T=0.05, seed=7,
                           sonda_rete=False)
    out = salva_report_griglie(righe, out_dir=str(tmp_path), T=0.05,
                               seed=7)
    assert os.path.isfile(out["csv"])


def test_diffusione_tutte_config():
    righe = valuta_diffusione(N=16, T=0.05, seed=7)
    assert len(righe) == 6
    nomi = [r["config"] for r in righe]
    assert "tutto-zero" in nomi and "base" in nomi
    for r in righe:
        assert r["finiti"]
        assert np.isfinite(r["Q"])


def test_diffusione_report_tmp(tmp_path):
    righe = valuta_diffusione(N=16, T=0.05, seed=7)
    out = salva_report_diffusione(righe, out_dir=str(tmp_path), N=16,
                                  T=0.05, seed=7)
    assert os.path.isfile(out["fig"])


def test_cue_tipi_presenti_e_report(tmp_path):
    assert set(TIPI_CUE) == {"multi-bump", "sparso", "random",
                             "striscia", "checker"}
    righe = valuta_cue_nonstrutturati(N=16, T=0.05, seed=7,
                                      n_per_tipo=3)
    assert len(righe) == 5
    for r in righe:
        assert 0.0 <= r["tasso_cert"] <= 1.0
    out = salva_report_cue(righe, out_dir=str(tmp_path), N=16,
                           T=0.05, seed=7)
    assert os.path.isfile(out["csv"])


def test_campi_naturali_massa_uno():
    for nome in CAMP_REALI:
        F = campo_naturale(nome, 16, seed=7)
        assert np.all(F >= 0.0)
        dx = 1.0 / 16
        assert abs(float(F.sum() * dx ** 2) - 1.0) < 1e-9


def test_reali_struttura_e_report(tmp_path):
    righe = valuta_reali(N=16, T=0.05, seed=7)
    assert len(righe) == 2  # rumore 0.0 e 0.5
    assert righe[0]["n_tot"] == len(CAMP_REALI)
    assert 0.0 <= righe[0]["acc"] <= 1.0
    out = salva_report_reali(righe, out_dir=str(tmp_path), N=16,
                             T=0.05, seed=7)
    assert os.path.isfile(out["md"])
