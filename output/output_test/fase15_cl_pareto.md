# Fase 15 — Pareto CL con shift di classe (stessa Q)

N=16 T=0.05 seed=7 shift=True

| modello | max degr | medio | distrutti | Q vecchie | Q nuove | costo |
|---|---|---|---|---|---|---|
| ingenua | 0.0462 | 0.0172 | 13/30 | 0.8350 | 0.8316 | 0.0 KB |
| replay_K10_B30 | 0.0402 | 0.0136 | 12/30 | 0.8387 | 0.8287 | 60.0 KB |
| replay_K10_B200 | 0.0312 | 0.0100 | 9/30 | 0.8422 | 0.8327 | 180.0 KB |
| replay_K20_B200 | 0.0299 | 0.0085 | 8/30 | 0.8437 | 0.8346 | 180.0 KB |
| ewc_l0.1 | 0.0239 | 0.0054 | 3/30 | 0.8468 | 0.8364 | 4.0 KB |
| ewc_l0.2 | 0.0172 | 0.0026 | 0/30 | 0.8496 | 0.8392 | 4.0 KB |
| ewc_l0.5 | 0.0104 | 0.0006 | 0/30 | 0.8516 | 0.8426 | 4.0 KB |
| ewc_l1 | 0.0081 | 0.0001 | 0/30 | 0.8521 | 0.8447 | 4.0 KB |

Protetta: max degr 0, distrutti 0 (strutturale). Costo protetta = 4 campi/slot (vedi plasticita').
