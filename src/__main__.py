"""Entry-point unificato: python -m src <2d|1d|demo|all> [opzioni]."""

from __future__ import annotations

import argparse
import sys


def cmd_2d(args):
    from .frammento_2d import Param, simula, riepilogo, invariante_nv

    p = Param(N=args.N, seed=args.seed)
    snap = simula(p, T=args.T, protocollo=args.protocollo)
    print(riepilogo(snap))
    for nome, D in (("g0", p.D0), ("gx", p.Dx), ("gy", p.Dy)):
        inv = invariante_nv(p, D)
        print(f"{nome:>2}: n = {inv['n']:.4f}  v = {inv['v']:.6f} "
              f" v~ = {inv['v_tilde']:.4f}  D_stimato = {inv['D_stimato']:.6f}"
              f" (vero {D:.6f})")


def cmd_1d(args):
    from .frammento_1d import Params, FrammentoDelVeloce, plot
    import os

    p = Params(T=args.T1d, N=args.N1d, gamma=args.gamma, beta=args.beta,
               novelty_budget=args.budget, conserve_mass=not args.no_mass,
               seed=args.seed)
    sim = FrammentoDelVeloce(p, evolve_g0=args.evolve_g0)
    sim.run()
    if args.plot:
        plot(sim, args.plot)
    else:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plot(sim, os.path.join(root, "output", "frammento_1d.png"))


def cmd_demo(args):
    from .demo_figure import main as demo_main
    demo_main()


def build_parser():
    ap = argparse.ArgumentParser(description="Frammento del veloce - runner unificato")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a2 = sub.add_parser("2d", help="simulazione 2D (modello principale)")
    a2.add_argument("--N", type=int, default=96)
    a2.add_argument("--T", type=float, default=0.30)
    a2.add_argument("--protocollo", default="stimolo", choices=["stimolo", "rilassamento"])
    a2.add_argument("--seed", type=int, default=7)
    a2.set_defaults(func=cmd_2d)

    a1 = sub.add_parser("1d", help="simulatore 1D di riferimento")
    a1.add_argument("--N1d", type=int, default=256)
    a1.add_argument("--T1d", type=float, default=2.0)
    a1.add_argument("--gamma", type=float, default=0.02)
    a1.add_argument("--beta", type=float, default=1.2)
    a1.add_argument("--budget", type=float, default=0.25)
    a1.add_argument("--no-mass", action="store_true")
    a1.add_argument("--evolve-g0", action="store_true")
    a1.add_argument("--seed", type=int, default=0)
    a1.add_argument("--plot", type=str, default="")
    a1.set_defaults(func=cmd_1d)

    ad = sub.add_parser("demo", help="7 figure dimostrative 2D in output/")
    ad.set_defaults(func=cmd_demo)

    aA = sub.add_parser("all", help="2d + 1d + demo in sequenza")
    aA.add_argument("--N", type=int, default=96)
    aA.add_argument("--T", type=float, default=0.30)
    aA.add_argument("--protocollo", default="stimolo", choices=["stimolo", "rilassamento"])
    aA.add_argument("--seed", type=int, default=7)
    aA.add_argument("--N1d", type=int, default=256)
    aA.add_argument("--T1d", type=float, default=2.0)
    aA.add_argument("--gamma", type=float, default=0.02)
    aA.add_argument("--beta", type=float, default=1.2)
    aA.add_argument("--budget", type=float, default=0.25)
    aA.add_argument("--no-mass", action="store_true")
    aA.add_argument("--evolve-g0", action="store_true")
    aA.add_argument("--plot", type=str, default="")
    def func_all(a):
        cmd_2d(a)
        # adatta namespace per cmd_1d
        cmd_1d(a)
        cmd_demo(a)
    aA.set_defaults(func=func_all)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
