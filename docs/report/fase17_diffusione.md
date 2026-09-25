# Fase 17.3 — Diffusione zero/ridotta (N=32 T=0.10 seed=7)

`ruvidezza` = media |lap(Fx)| finale (stati spigolosi se alta). Con D=0 il dt e' fallback reattivo 5e-4 (dt_stabile non definito); vedi `_dt_sicuro`.

| config | D0/Dx/Dy | Q | eq | rnov | cert | ruvidezza |
|---|---|---|---|---|---|---|
| base | 0.050/0.010/0.002 | 0.593 | 0.389 | 0.006 | SI | 158.74 |
| D0=0 | 0.000/0.010/0.002 | 0.608 | 0.392 | 0.006 | NO | 164.87 |
| Dx=0 | 0.050/0.000/0.002 | 0.732 | 0.641 | 0.006 | NO | 227.41 |
| Dy=0 | 0.050/0.010/0.000 | 0.593 | 0.389 | 0.007 | SI | 158.74 |
| tutto-zero | 0.000/0.000/0.000 | 0.749 | 0.251 | 0.007 | NO | 236.41 |
| quasi-zero | 0.001/0.001/0.001 | 0.732 | 0.252 | 0.006 | NO | 225.83 |

Motivi di mancata certificazione:

- D0=0: non certificato: quiete inattiva: Lyapunov non monotona fraz_dV<=0=0.020<0.9
- Dx=0: non certificato: quiete inattiva: operativita' non allineata err_fx_fo=0.6410>=eps=0.6
- tutto-zero: non certificato: quiete inattiva: Lyapunov non monotona fraz_dV<=0=0.005<0.9
- quasi-zero: non certificato: quiete inattiva: Lyapunov non monotona fraz_dV<=0=0.500<0.9

Lettura: se Q/cert crollano senza diffusione, la diffusione e' essenziale; se tengono, e' solo conveniente.
