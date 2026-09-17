# Ablazione dei livelli

| config | massa_g0_fin | qualita_fin | continuita_media | fedelta | max_dVdt | fraz_dVdt_nonpos | novita_fin | t_rec | recuperato | adattamento |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 solo diffusione | 1.0000 | 0.7088 | 0.9996 | 1.0000 | -5.5671 | 1.0000 | 0.0119 | 0.3000 | no | -0.0405 |
| 2 g0+gx (senza gy) | 1.0000 | 0.4877 | 0.9995 | 1.0000 | -6.8289 | 1.0000 | 0.0119 | 0.3000 | no | -0.0351 |
| 3 completo | 1.0000 | 0.4877 | 0.9995 | 1.0000 | -6.8289 | 1.0000 | 0.0121 | 0.3000 | no | -0.0351 |
| 4 completo senza vincolo (ky=0, K=5) | 1.0000 | 0.4877 | 0.9995 | 1.0000 | -6.8289 | 1.0000 | 0.0125 | 0.3000 | no | -0.0351 |
| 5 completo senza rumore | 1.0000 | 0.4877 | 0.9995 | 1.0000 | -6.8289 | 1.0000 | 0.0121 | 0.3000 | no | -0.0351 |
| 6 rumore eccessivo (gamma=0.25) | 1.0000 | 0.4877 | 0.9995 | 0.9998 | -6.8289 | 1.0000 | 0.0121 | 0.3000 | no | -0.0351 |
