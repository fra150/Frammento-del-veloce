# Fase 17.1 — Tempi lunghi (deterministico, N=48 seed=7)

Una run per T. Atteso: dV/dt<=0 resta ~1.0 (strutturale), Fo diffonde verso il piatto quindi ea cresce e la quiete tarata sul regime breve puo' cadere a T grandi (risultato onesto, non bug).

| T | dV<=0 | V0->Vfine | drift g0 | eq | ea | Q | rnov | quiete | cert |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 1.000 | 4.094->0.249 | 0.0e+00 | 0.653 | 0.872 | 0.113 | 0.005 | NO | NO |
| 1.00 | 1.000 | 4.094->0.202 | 0.0e+00 | 0.789 | 0.924 | 0.033 | 0.005 | NO | NO |
| 2.00 | 0.869 | 4.094->0.199 | 0.0e+00 | 0.814 | 0.943 | 0.021 | 0.007 | NO | NO |
| 5.00 | 0.669 | 4.094->0.199 | -2.2e-16 | 0.814 | 0.945 | 0.019 | 0.013 | NO | NO |

Limite: seed singolo, protocollo stimolo; lo stimolo Lissajous ha periodi 0.30/0.45 quindi a T>>0.45 la media e' su molti cicli.
