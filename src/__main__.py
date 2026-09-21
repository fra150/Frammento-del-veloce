"""Entry-point unificato: python -m src <2d|1d|demo|sweep|ablazione|all>."""

from __future__ import annotations

import argparse
import sys


def cmd_2d(args):
    from .frammento_2d import (Param, simula, riepilogo, invariante_nv,
                               verifica_invarianza)

    p = Param(N=args.N, seed=args.seed)
    snap = simula(p, T=args.T, protocollo=args.protocollo)
    print(riepilogo(snap))
    for nome, D in (("g0", p.D0), ("gx", p.Dx), ("gy", p.Dy)):
        inv = invariante_nv(p, D)
        print(f"{nome:>2}: n = {inv['n']:.4f}  v = {inv['v']:.6f} "
              f" v~ = {inv['v_tilde']:.4f}  D_stimato = {inv['D_stimato']:.6f}"
              f" (vero {D:.6f})  lambda_g = {inv['lambda_g']:.4f}")
    ver = verifica_invarianza(p)
    lam = ver["lambda"]
    print(f"stimatore diffusivita': lambda_g g0/gx/gy = {lam['g0']:.3f} / "
          f"{lam['gx']:.3f} / {lam['gy']:.3f}  media = {ver['media']:.3f} ± "
          f"{ver['std']:.3f} (CV = {ver['cv'] * 100:.1f}%, "
          f"D in [{ver['d_min']:.4f}, {ver['d_max']:.4f}])")


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


def cmd_sweep(args):
    from .studi import main_sweep
    import os
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    seeds = tuple(args.seeds) if getattr(args, "seeds", None) else None
    main_sweep(N=args.N, T=args.T, out_dir=out, seed=args.seed, seeds=seeds)


def cmd_ablazione(args):
    from .studi import main_ablazione
    import os
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    seeds = tuple(args.seeds) if getattr(args, "seeds", None) else None
    main_ablazione(N=args.N, T=args.T, out_dir=out, seed=args.seed,
                   seeds=seeds)


def cmd_gf(args):
    from .frammento_2d import Param, simula
    from .frammento_gf import verifica_quiete, certifica_frammento, MemoriaGF, diagnostica_gf

    p = Param(N=args.N, seed=args.seed)
    snap = simula(p, T=args.T, protocollo=args.protocollo, salva_ogni=20)
    Fo, Fx, Fy, ess = snap["F0"][-1], snap["Fx"][-1], snap["Fy"][-1], snap["essenza"]
    q = verifica_quiete(Fo, Fx, ess, p.dx, Fy=Fy, V_hist=snap["diag"]["V"])
    c = certifica_frammento(Fo, Fx, Fy, ess, p.dx, diag=snap["diag"])
    print(diagnostica_gf(q, c))
    print(f"motivo: {c['motivo']}")
    # demo memoria a costo zero: salva + doppio richiamo
    mem = MemoriaGF()
    k = mem.chiave(Fo, Fx, Fy, ess, dx=p.dx)
    if c["certificato"]:
        mem.salva(k, {"qualita": c["qualita"], "quiete": q})
        hit1, _ = mem.richiama(k)
        hit2, _ = mem.richiama(k)
        print(f"cache: chiave {k[:12]}... salva->richiama hit1={hit1} hit2={hit2} (costo ricomputazione ~0) stats={mem.stats()}")
    else:
        print(f"cache: non salvato (non certificato) stats={mem.stats()}")


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

    aS = sub.add_parser("sweep", help="studio di robustezza al variare dei parametri")
    aS.add_argument("--N", type=int, default=48)
    aS.add_argument("--T", type=float, default=0.15)
    aS.add_argument("--seed", type=int, default=7)
    aS.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="se multipli, sweep multi-seed con media±std "
                         "(es. --seeds 7 11 13 21 33)")
    aS.add_argument("--out", type=str, default="")
    aS.set_defaults(func=cmd_sweep)

    aB = sub.add_parser("ablazione", help="confronto con baseline di controllo")
    aB.add_argument("--N", type=int, default=48)
    aB.add_argument("--T", type=float, default=0.15)
    aB.add_argument("--seed", type=int, default=7)
    aB.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="se multipli, ablazione multi-seed con media±std")
    aB.add_argument("--out", type=str, default="")
    aB.set_defaults(func=cmd_ablazione)

    aG = sub.add_parser("gf", help="quiete attiva + certificazione + cache (livello gf)")
    aG.add_argument("--N", type=int, default=48)
    aG.add_argument("--T", type=float, default=0.15)
    aG.add_argument("--protocollo", default="stimolo", choices=["stimolo", "rilassamento"])
    aG.add_argument("--seed", type=int, default=7)
    aG.set_defaults(func=cmd_gf)

    def cmd_stress500(args):
        from .stress_500 import OUT_DIR_DEFAULT, esegui
        esegui(n=args.n, N=args.N, T=args.T, seed=args.seed,
               out_dir=args.out or OUT_DIR_DEFAULT)

    def cmd_rete1000(args):
        from .rete_frammento import OUT_DIR_DEFAULT, esegui_test_1000, salva_report_r1000
        res = esegui_test_1000(n_cert=args.n_cert, n_nuove=args.n_nuove,
                               N=args.N, T=args.T, seed=args.seed,
                               capacita_max=args.capacita_max,
                               politica=args.politica)
        salva_report_r1000(res, out_dir=args.out or OUT_DIR_DEFAULT,
                           N=args.N, T=args.T, seed=args.seed)

    def cmd_assoc(args):
        from .rete_frammento import OUT_DIR_DEFAULT, esegui_test_associativo, salva_report_assoc
        res = esegui_test_associativo(n_cert=args.n_cert, n_nuove=args.n_nuove,
                                      N=args.N, T=args.T, seed=args.seed,
                                      tipo=args.tipo)
        salva_report_assoc(res, out_dir=args.out or OUT_DIR_DEFAULT,
                           N=args.N, T=args.T, seed=args.seed)

    aS5 = sub.add_parser("stress500", help="stress test N domande g0->gf + figure")
    aS5.add_argument("--n", type=int, default=500)
    aS5.add_argument("--N", type=int, default=32)
    aS5.add_argument("--T", type=float, default=0.10)
    aS5.add_argument("--seed", type=int, default=7)
    aS5.add_argument("--out", type=str, default="")
    aS5.set_defaults(func=cmd_stress500)

    aR = sub.add_parser("rete1000", help="test dei 1000: interferenza retroattiva (rete che non distrugge)")
    aR.add_argument("--n-cert", type=int, default=200)
    aR.add_argument("--n-nuove", type=int, default=800)
    aR.add_argument("--N", type=int, default=32)
    aR.add_argument("--T", type=float, default=0.10)
    aR.add_argument("--seed", type=int, default=7)
    aR.add_argument("--capacita-max", type=int, default=None)
    aR.add_argument("--politica", default="espandi", choices=["espandi", "rifiuta"])
    aR.add_argument("--out", type=str, default="")
    aR.set_defaults(func=cmd_rete1000)

    aAs = sub.add_parser("assoc", help="richiamo associativo: cue parziali -> ricostruzione + shift di classe")
    aAs.add_argument("--n-cert", type=int, default=200)
    aAs.add_argument("--n-nuove", type=int, default=800)
    aAs.add_argument("--N", type=int, default=32)
    aAs.add_argument("--T", type=float, default=0.10)
    aAs.add_argument("--seed", type=int, default=7)
    aAs.add_argument("--tipo", default="blocco", choices=["blocco", "casuale"])
    aAs.add_argument("--out", type=str, default="")
    aAs.set_defaults(func=cmd_assoc)

    def cmd_fase15(args):
        from .fase15 import (sweep_pareto_cl, salva_report_cl,
                             misura_plasticita, salva_report_plasticita,
                             test_ood_mix_retrieval,
                             test_ood_rumore_retrieval, salva_report_ood)
        out = args.out or None
        righe = sweep_pareto_cl(n_cert=args.n_cert, n_nuove=args.n_nuove,
                                N=args.N, T=args.T, seed=args.seed,
                                shift=True)
        salva_report_cl(righe, out_dir=out, N=args.N, T=args.T,
                        seed=args.seed, shift=True)
        plast = misura_plasticita(n_list=[50, 100, 200, 400, 800],
                                  N=args.N, T=args.T, seed=args.seed)
        salva_report_plasticita(plast, out_dir=out, N=args.N, T=args.T,
                                seed=args.seed)
        mix = test_ood_mix_retrieval(N=args.N, T=args.T, seed=args.seed)
        rum = test_ood_rumore_retrieval(N=args.N, T=args.T, seed=args.seed)
        salva_report_ood(mix, rum, out_dir=out, N=args.N, T=args.T,
                         seed=args.seed)

    aF15 = sub.add_parser("fase15", help="Pareto CL + plasticita' + OOD (referee)")
    aF15.add_argument("--n-cert", type=int, default=30)
    aF15.add_argument("--n-nuove", type=int, default=60)
    aF15.add_argument("--N", type=int, default=16)
    aF15.add_argument("--T", type=float, default=0.05)
    aF15.add_argument("--seed", type=int, default=7)
    aF15.add_argument("--out", type=str, default="")
    aF15.set_defaults(func=cmd_fase15)

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
