# Fase 16 — Transfer standard (BWT/FWT, 2 task A->B)

A=30 B=60 N=16 T=0.05 seed=7

| modello | R_AA | R_AB | R_BB | R_B0 | BWT | FWT |
|---|---|---|---|---|---|---|
| protetta | 0.8522 | 0.8522 | 0.8494 | 0.8494 | 0.0000 | 0.0000 |
| ingenua | 0.8309 | 0.8350 | 0.8316 | 0.8335 | 0.0041 | -0.0019 |
| replay | 0.8357 | 0.8422 | 0.8327 | 0.8370 | 0.0065 | -0.0043 |
| ewc | 0.8472 | 0.8516 | 0.8426 | 0.8447 | 0.0044 | -0.0020 |

BWT<0 = forgetting; FWT>0 = A aiuta B. Protetta attesa BWT=FWT=0 (isolamento, non transfer).
