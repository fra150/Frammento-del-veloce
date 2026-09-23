# Fase 16-bis — Transfer v2 (fedelta' + zero-shot)

A=30 B=60 N=16 T=0.05 seed=7

| modello | F_AA | F_AB | BWT_fid | BWT_Q | FWT_zero |
|---|---|---|---|---|---|
| protetta | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8440 |
| ingenua | 0.9553 | 0.8974 | -0.0579 | 0.0041 | 0.8415 |
| replay | 0.9670 | 0.9194 | -0.0476 | 0.0065 | 0.8458 |
| ewc | 0.9871 | 0.9519 | -0.0351 | 0.0044 | 0.8561 |

BWT_fid<0 = forgetting vero (vs certificato); BWT_Q ~0 conferma che la media-Q non lo vede. FWT_zero = generalizzazione in avanti su cue mai viste (qui i condivisi possono vincere: seconda faccia del trade-off).
