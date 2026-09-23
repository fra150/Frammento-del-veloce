# Fase 15 � Plasticita' (capacita' vs Q, memoria vs slot)

N=16 T=0.05 seed=7

| n richieste | certificati | tasso | Q media | ms/ricordo | memoria |
|---|---|---|---|---|---|
| 50 | 50 | 1.000 | 0.9479 | 2.13 | 400 KB |
| 100 | 100 | 1.000 | 0.9483 | 1.95 | 800 KB |
| 200 | 200 | 1.000 | 0.9492 | 1.87 | 1600 KB |
| 400 | 400 | 1.000 | 0.9488 | 1.87 | 3200 KB |
| 800 | 800 | 1.000 | 0.9488 | 1.96 | 6400 KB |

Fattura dello zero-forgetting strutturale: memoria lineare 4 campi float64/slot, tempo lineare per ricordo.

## Regime a memoria limitata (politica `rifiuta`, N=16 T=0.05 seed=7, 800 richieste)

| capacita_max | certificabili | memorizzati | copertura | Q media slot |
|---|---|---|---|---|
| 100 | 800 | 100 | 0.125 | 0.9483 |
| 200 | 800 | 200 | 0.250 | 0.9492 |

Lettura onesta ("gabbia dorata" quantificata): la Q dei memorizzati non
degrada mai (piatta ~0.949), ma a memoria limitata la copertura crolla
(12.5% con cap 100 su 800). Il gate gf in regime facile non filtra
(tasso 1.0 anche con bump 3-8); la saturazione emerge solo come
rifiuto per capacita'. Espandere = pagare memoria lineare.
