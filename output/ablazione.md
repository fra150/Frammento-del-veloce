# Ablazione dei livelli

| config | massa_g0_fin | qualita_fin | continuita_media | fedelta | max_dVdt | fraz_dVdt_nonpos | novita_fin | t_rec | recuperato | adattamento |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 solo diffusione | 1.0000 | 0.7088 | 0.9996 | 1.0000 | -5.5671 | 1.0000 | 0.0119 | 1.0000 | no | 0.2040 |
| 2 g0+gx (senza gy) | 1.0000 | 0.4877 | 0.9995 | 1.0000 | -6.8289 | 1.0000 | 0.0119 | 0.4905 | si | 0.4793 |
| 3 completo | 1.0000 | 0.4877 | 0.9995 | 0.9986 | -6.8289 | 1.0000 | 0.0121 | 0.4905 | si | 0.4793 |
| 4 completo senza vincolo (ky=0, K=5) | 1.0000 | 0.4877 | 0.9995 | 0.9987 | -6.8289 | 1.0000 | 0.0125 | 0.4905 | si | 0.4793 |
| 5 completo senza rumore | 1.0000 | 0.4877 | 0.9995 | 1.0000 | -6.8289 | 1.0000 | 0.0121 | 0.4905 | si | 0.4793 |
| 6 rumore eccessivo (gamma=0.25) | 1.0000 | 0.4877 | 0.9995 | 0.8924 | -6.8287 | 1.0000 | 0.0128 | 0.4905 | si | 0.4793 |
