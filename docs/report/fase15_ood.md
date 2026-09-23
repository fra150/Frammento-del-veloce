# Fase 15 — OOD con retrieval (prior non oracolo)

N=16 T=0.05 seed=7

## Mix A+B (alpha=1 -> puro A, 0 -> puro B)

| alpha | vincitore | margine | q vs mix | q vs A | q vs B |
|---|---|---|---|---|---|
| 0.00 | B | 4.36e+07 | 0.8324 | 0.8534 | 0.8324 |
| 0.25 | B | 2.14e-01 | 0.8623 | 0.8538 | 0.8305 |
| 0.50 | B | 1.59e-01 | 0.8775 | 0.8540 | 0.8285 |
| 0.75 | A | 6.53e-04 | 0.8738 | 0.8541 | 0.8264 |
| 1.00 | A | 1.44e+05 | 0.8542 | 0.8542 | 0.8243 |

## Rumore (retrieval dello stesso ricordo)

| rumore | accuratezza | margine medio |
|---|---|---|
| 0.00 | 1.000 | 1.40e+09 |
| 0.50 | 0.250 | 5.49e-03 |
| 1.00 | 0.125 | 2.87e-03 |
| 2.00 | 0.125 | 1.47e-03 |

Lettura: margine ~1e-6 = decisione ambigua (campi ~97% simili, il visibile non discrimina). Il completamento ricostruisce, non ragiona: il prior recuperato vince.
