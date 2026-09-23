# Fase 16 — Sonno (gist offline, slot intatti)

N=16 T=0.05 seed=7 soglia_margine=0.05

Politica: se margine < soglia usa il gist, altrimenti il vincitore. Il sonno non scrive mai negli slot.

## Mix A+B

| alpha | vincitore | margine | q winner | q gist | q scelta |
|---|---|---|---|---|---|
| 0.00 | B | 4.36e+07 | 0.8324 | 0.8707 | 0.8324 |
| 0.25 | B | 2.14e-01 | 0.8623 | 0.9092 | 0.8623 |
| 0.50 | B | 1.59e-01 | 0.8775 | 0.9296 | 0.8775 |
| 0.75 | A | 6.53e-04 | 0.8738 | 0.9190 | 0.9190 |
| 1.00 | A | 1.44e+05 | 0.8542 | 0.8878 | 0.8542 |

## Rumore

| rumore | acc winner | q winner | q gist | fraz gist |
|---|---|---|---|---|
| 0.00 | 1.000 | 0.8418 | 0.8655 | 0.00 |
| 0.50 | 0.250 | 0.8105 | 0.8262 | 1.00 |
| 1.00 | 0.125 | 0.7303 | 0.7389 | 1.00 |
| 2.00 | 0.125 | 0.5262 | 0.5288 | 1.00 |

Lettura: il gist e' pooling sfocato; aiuta solo dove il winner e' ambiguo. Se q_gist <= q_winner ovunque, il sonno non risolve l'OOD (negativo onesto).
