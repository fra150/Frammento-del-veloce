"""Test CLI unificata e import demo (src/__main__.py, src/demo_figure.py)."""

from src.__main__ import build_parser


def test_parser_sottocomandi():
    ap = build_parser()
    a = ap.parse_args(["2d", "--N", "32", "--T", "0.01"])
    assert a.cmd == "2d" and a.N == 32
    a = ap.parse_args(["1d", "--N1d", "64", "--T1d", "0.05"])
    assert a.cmd == "1d" and a.N1d == 64
    a = ap.parse_args(["demo"])
    assert a.cmd == "demo"
    a = ap.parse_args(["all"])
    assert a.cmd == "all"


def test_demo_funzioni_importabili():
    import src.demo_figure as demo

    for nome in (
        "fig_tre_livelli",
        "fig_evoluzione",
        "fig_diagnostica",
        "fig_metriche",
        "fig_turing",
        "fig_invariante",
        "fig_lfp",
        "main",
    ):
        assert callable(getattr(demo, nome)), nome


def test_cmd_2d_esegue(capsys):
    ap = build_parser()
    a = ap.parse_args(["2d", "--N", "64", "--T", "0.05", "--seed", "7"])
    a.func(a)
    assert "FRAMMENTO DEL VELOCE" in capsys.readouterr().out


def test_cmd_1d_esegue(tmp_path):
    ap = build_parser()
    plot_path = str(tmp_path / "prova_1d.png")
    a = ap.parse_args(["1d", "--N1d", "32", "--T1d", "0.5",
                       "--seed", "0", "--plot", plot_path])
    a.func(a)
    import os
    assert os.path.isfile(plot_path)


def test_package_export():
    import src

    assert src.Param is not None
    assert src.FrammentoDelVeloce is not None
