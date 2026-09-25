# Fase 17.2 — Griglie enormi (T=0.050 det., seed=7)

Costo ~N^4 a T fisso (passi ~N^2, costo/passo ~N^2). Turing solo N<=64 (dt fisso instabile oltre). Sonda rete (3 cue) solo N<=128.

| N | dt | passi | wall s | dV<=0 | drift g0 | Q | cert | mem/slot | sonda |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 2.0e-03 | 25 | 0.0 | 1.000 | 0.0e+00 | 0.768 | SI | 32 KB | 1.00 |
| 64 | 4.9e-04 | 102 | 0.1 | 1.000 | -1.1e-16 | 0.763 | SI | 128 KB | 1.00 |
| 128 | 1.2e-04 | 409 | 0.4 | 1.000 | 0.0e+00 | 0.762 | SI | 512 KB | 1.00 |
| 256 | 3.1e-05 | 1638 | 8.3 | 1.000 | 1.1e-16 | 0.762 | SI | 2048 KB | n/d |
| 512 | 7.6e-06 | 6553 | 665.5 | 1.000 | 2.2e-16 | 0.762 | SI | 8192 KB | n/d |

Limite: seed singolo; N=512 solo su richiesta (--include-512) per costo.
