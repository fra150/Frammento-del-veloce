# Fase 17.5 — Dati naturali (N=32 T=0.10 seed=7)

Campi dimostrativi NON gaussiani, normalizzati a massa 1. Sono stand-in onesti (non claim di validazione reale): il protocollo accetta anche PNG/WAV/CSV veri convertiti allo stesso formato (array >=0, massa 1, lato N).

| campo | cert | Q | eq | rnov |
|---|---|---|---|---|
| gradiente | NO | 0.074 | 0.804 | 0.006 |
| barre | NO | 0.079 | 0.789 | 0.006 |
| chirp | NO | 0.077 | 0.794 | 0.006 |
| ripple | NO | 0.077 | 0.788 | 0.006 |
| multiscala | NO | 0.081 | 0.777 | 0.006 |
| eeg-sintetico | NO | 0.079 | 0.787 | 0.006 |

Motivi di mancata certificazione:

- gradiente: non certificato: quiete inattiva: operativita' non allineata err_fx_fo=0.8040>=eps=0.6
- barre: non certificato: quiete inattiva: operativita' non allineata err_fx_fo=0.7889>=eps=0.6
- chirp: non certificato: quiete inattiva: operativita' non allineata err_fx_fo=0.7943>=eps=0.6
- ripple: non certificato: quiete inattiva: operativita' non allineata err_fx_fo=0.7878>=eps=0.6
- multiscala: non certificato: quiete inattiva: operativita' non allineata err_fx_fo=0.7774>=eps=0.6
- eeg-sintetico: non certificato: quiete inattiva: operativita' non allineata err_fx_fo=0.7869>=eps=0.6

## Retrieval nearest-MSE sui certificati

| rumore | n_cert/n_tot | acc | qmask |
|---|---|---|---|
| 0.0 | 0/6 | 0.000 | 0.000 |
| 0.5 | 0/6 | 0.000 | 0.000 |

Limite: mapping immagine/audio->campo e' arbitrario; EEG veri restano il pilota nullo di Fase 12.
