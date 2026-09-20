# Richiamo associativo da cue parziali (ricostruzione)

N=32 T=0.10 seed=7 tipo=blocco rumore=0.0 T_rec=5.0 amp bump=(3.0, 8.0)
shift di classe: sinistra -> destra (||P_B-P_A||/||P_A|| = 0.177)
certificati A: 200 | nuove apprese B: 800 | nucleo ok=True
verifica protetta (f=0.5, prima vs dopo B): max diff = 0.00e+00

## Operatori di completamento (protetta, prior = core)

| frazione | frac eff | residuo | media | core | armonica | gx (canonico) |
|---|---|---|---|---|---|---|
| 0.25 | 0.250 | 0.7139±0.0210 | 0.0571 | 0.6553 | 0.4729 | 0.5132 |
| 0.50 | 0.517 | 0.6491±0.0221 | 0.0099 | 0.6531 | 0.0499 | 0.3318 |
| 0.75 | 0.766 | 0.6528±0.0173 | 0.0027 | 0.6415 | 0.0062 | 0.3376 |

## Degrado in regime associativo (protetta vs condivisa)

| frazione | protetta | condivisa P_A (controllo) | condivisa dopo shift | degrado medio | degrado max | >0.05 |
|---|---|---|---|---|---|---|
| 0.25 | 0.7139±0.0210 | 0.9472 | 0.9459±0.0401 | 0.0013 | 0.0103 | 0% |
| 0.50 | 0.6491±0.0221 | 0.9104 | 0.8865±0.0302 | 0.0240 | 0.0507 | 1% |
| 0.75 | 0.6528±0.0173 | 0.9008 | 0.8589±0.0239 | 0.0420 | 0.0679 | 28% |

Protetta: ricostruzione deterministica con core congelato; degrado esattamente 0 (verificato bit-identico).
Condivisa: prior = campo condiviso; dentro la stessa classe il pooling aiuta (qualita' assoluta piu' alta), ma dopo lo shift il prior segue i dati nuovi e i vecchi ricordi degradano.
- figura: fig13_assoc.png
