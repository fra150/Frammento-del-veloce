# Fase 12 — gruppo 10 soggetti ds005095 (ICA + trend + permutazione, locale)

Data: 19/09/2026. Soggetti sub-04..sub-13, ses-01, pool frontale Fz/F3/F4,
fs 250 Hz, passa-alto 1 Hz + ICA FastICA 15 comp con ricerca EOG su Fp1
(0 componenti escluse in tutti i soggetti: nessun blink catturato dal
detector, resta il passa-alto contro la deriva).
Finestra retention 1.5–3.5 s, bande theta 4–8 / gamma 30–48 (no 50 Hz),
metrica g/t per trial → mediana per carico. Dati grezzi ~3.1 GB in Temp,
mai nel repo. DOI 10.18112/openneuro.ds005095.v1.0.2, licenza CC0.

## Mediane g/t per soggetto × carico

| sub | 3 | 6 | 9 | 12 | 15 |
|---|---|---|---|---|---|
| 04 | 1.768 | 1.399 | 1.395 | 1.564 | 1.486 |
| 05 | 0.331 | 0.298 | 0.233 | 0.305 | 0.418 |
| 06 | 0.288 | 0.362 | 0.276 | 0.275 | 0.345 |
| 07 | 0.199 | 0.132 | 0.103 | 0.139 | 0.118 |
| 08 | 1.216 | 0.697 | 0.490 | 0.334 | 0.606 |
| 09 | 0.459 | 0.352 | 0.367 | 0.448 | 0.449 |
| 10 | 0.588 | 0.427 | 0.641 | 0.450 | 0.715 |
| 11 | 0.279 | 0.431 | 0.299 | 0.374 | 0.116 |
| 12 | 0.643 | 1.030 | 0.521 | 0.584 | 0.594 |
| 13 | 0.331 | 0.385 | 0.334 | 0.383 | 0.420 |

Medie di gruppo: 0.610, 0.551, 0.466, 0.486, 0.527.

## Test

- Spearman pooled carico vs g/t: **rho=-0.009, p=0.95** (zero).
- Friedman sui 5 carichi: chi2=8.32, **p=0.081** (n.s.).
- Permutazione entro-soggetto (2000, two-sided su |rho|): **p=0.89**.

## Lettura

1. Nessun trend carico→gamma/theta: né monotonically crescente (predizione
   giocattolo), né decrescente. Piatto con rumore.
2. Variabilità inter-soggetto ~10x (sub-07 ~0.13 vs sub-04 ~1.5): con questa
   metrica grezza il soggetto domina sul carico — altro motivo per cui il
   pilota N=1 non significava nulla.
3. ICA non ha agganciato blink (Fp1 detector vuoto): la pulizia reale
   richiederebbe revisione manuale componenti + rigetto muscolare. Anche con
   quella, un effetto assente a rho=-0.009 non diventa significativo.
4. Verdetto Fase 12: **nullo di gruppo**. Il modello resta astratto per
   scelta (README §7 corretto); `confronto_bio` resta ponte onesto senza
   claim. Se mai si vorrà riprovare: potenze per banda separate (non
   rapporto), baseline sottratta per trial, sorgenti/laplaciano, preregistrazione.
