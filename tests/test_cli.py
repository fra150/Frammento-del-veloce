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


def test_package_export():
    import src

    assert src.Param is not None
    assert src.FrammentoDelVeloce is not None
