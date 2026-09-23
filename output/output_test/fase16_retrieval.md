# Fase 16 — Retrieval repair (validazione + coarse-to-fine)

N=16 T=0.05 seed=7 k=3 fraz_val=0.2

| rumore | base | validato | coarse-to-fine |
|---|---|---|---|
| 0.00 | 1.000 | 0.250 | 0.250 |
| 0.50 | 0.250 | 0.250 | 0.250 |
| 1.00 | 0.125 | 0.250 | 0.250 |
| 2.00 | 0.125 | 0.250 | 0.250 |

Gara tra prior solo sul visibile (fit->val): mai l'occulto. Se validato <= base, il negativo OOD resta (onesto).
