# Fase 16-bis — Retrieval v2 (il 3% sceglie, il 97% resta)

N=16 T=0.05 seed=7 scelta=MSE pesata su varianza

| rumore | acc base | acc v2 | q base | q v2 | fraz gist |
|---|---|---|---|---|---|
| 0.00 | 1.000 | 1.000 | 0.8418 | 0.8418 | 0.00 |
| 0.50 | 0.250 | 0.375 | 0.8105 | 0.8230 | 0.50 |
| 1.00 | 0.125 | 0.375 | 0.7303 | 0.7409 | 0.50 |
| 2.00 | 0.125 | 0.375 | 0.5262 | 0.5326 | 0.62 |

Doppio criterio: acc_v2(0)==1.0 -> SI; acc_v2>base da qualche parte -> SI; guardia q_v2>=q_base ovunque -> SI.
**Verdetto: PROMOSSO** (ricostruzione sempre a campo intero col prior certificato).
