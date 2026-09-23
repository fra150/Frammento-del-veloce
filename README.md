# Frammento del Veloce

![coverage](https://img.shields.io/badge/coverage-83%25-brightgreen)
![tests](https://img.shields.io/badge/tests-108_passed-brightgreen)
![python](https://img.shields.io/badge/python-3.13-blue)
![CI](https://github.com/fra150/Frammento-del-veloce/actions/workflows/ci.yml/badge.svg)

Libreria di simulazione numerica per la diffusione dinamica della memoria.

> **Natura del modello.** Questo è un **modello computazionale astratto**,
> ispirato a proprietà note della memoria dinamica (consolidamento,
> riconsolidamento, interferenza, generazione di associazioni). Non è una
> descrizione validata della memoria biologica: le tracce theta/gamma e
> l'LFP sintetico sono **analogie computazionali**, non validazione
> neuroscientifica (per quella servirebbero dati EEG/fMRI/comportamentali).
> Vedi §7 Limiti del modello.

Modello a tre livelli geometrici:

- **g0** — essenza perfetta / invariante (`R = 0`, solo diffusione conservativa)
- **gx** — operativita' vincolata agli input (accoppiamento con `x(t)`)
- **gy** — novita' controllata (reazione non lineare + rumore, vincolata da `g0` e `gx`)

Livello trasversale **gf** (garanzia del frammento, `src/frammento_gf.py`):
quiete attiva + certificazione + memoria a costo zero. Non aggiunge dinamica,
verifica lo stato finale (`Fx` in quiete rispetto a `Fo`/essenza) e certifica
solo se quiete AND qualita' AND budget novita' sono ok (mai se quiete=False).

Il progetto distingue tre piani (§2): **modello concettuale** (l'idea di
g0/gx/gy), **modello matematico** (equazioni, vincoli, condizioni) e
**implementazione numerica** (codice, discretizzazione, test). Una simulazione
che funziona dimostra che il modello è implementato coerentemente e che, per
certi parametri, produce il comportamento previsto — non, da sola, che il
modello matematico sia valido in generale.

Autore della ricerca: **dr. Bulla Francesco** (Catania, 17/09/2026).
Uso didattico / scientifico.

Documentazione teorica completa: `frammetoveloce.md`.
Preprint PDF: `Frammento_del_veloce_IT.pdf`.

---

## 1. Struttura del progetto

```text
Framento del veloce/
├── src/
│   ├── __init__.py        # export unificati Param / Params + gf + bio
│   ├── __main__.py        # CLI: 2d | 1d | demo | sweep | ablazione | gf | stress500 | rete1000 | assoc | fase15 | fase16 | all
│   ├── frammento_2d.py    # modello 2D toroidale (codice principale)
│   ├── frammento_1d.py    # simulatore 1D di riferimento
│   ├── frammento_gf.py    # livello gf: quiete + certificazione + memoria
│   ├── rete_frammento.py  # rete che non distrugge: nucleo frozen + slot + test 1000 + richiamo associativo
│   ├── fase15.py          # prove referee: baseline CL (Replay/EWC-lite) + plasticita' + retrieval OOD
│   ├── fase16.py          # sonno (gist) + eviction + transfer BWT/FWT + retrieval repair
│   ├── confronto_bio.py   # coerenza LFP/theta-gamma + confronto spettrale onesto
│   ├── stress_500.py      # stress test N domande g0->gf + figure
│   ├── demo_figure.py     # genera le 7 figure del preprint
│   └── studi.py           # sweep parametri + ablazione (CSV, md, fig08)
├── tests/                 # 108 test (102 fast + 6 slow con --run-slow)
│   ├── test_gf.py         # 7 test quiete/certificazione/cache/correzione
│   ├── test_rete_1000.py  # 9 test rete che non distrugge (8 fast + 1 slow full-1000)
│   ├── test_rete_assoc.py # 8 test richiamo associativo (7 fast + 1 slow shift di classe)
│   ├── test_fase15.py     # 11 test CL/shift + plasticita' + retrieval OOD (fast)
│   ├── test_fase16.py     # 12 test sonno/eviction/transfer/retrieval repair (fast)
│   ├── test_confronto_bio.py  # 8 test coerenza LFP/PAC/confronto onesto
│   ├── test_bio_fase12.py # 3 test pipeline trend/permutazione (sintetico + matrice reale)
│   ├── test_stress_500.py # 3 test catena g0->gf + replica cache
│   └── ...
├── output/                # PNG/CSV/md generati (ignorati, TRANNE output_test versionato)
│   └── output_test/       # stress 500 + rete 1000 + assoc + fase15 + fase16: CSV + pannelli + md (push su GitHub)
├── .github/workflows/     # CI GitHub Actions (test + coverage)
├── run.py                 # avvio rapido: python run.py [all]
├── pyproject.toml         # marker slow + config coverage
├── requirements.txt
├── LICENSE                # MIT
├── CITATION.cff           # citazione + metadati Zenodo
└── README.md
```

| Modulo | Contenuto |
|---|---|
| `frammento_2d.py` | `Param`, `griglia`, `laplaciano`, `essenza`, `input_field` (Lissajous), `simula` (Eulero-Maruyama con `sqrt(dt)`), `lyapunov`, `derivata_numerica`, `esperimento_diffusione`, `invariante_nv`, `verifica_invarianza` (`lambda_g`), `turing_gy` (Gierer-Meinhardt), `metriche`, `fedelta`, `lfp_sintetico`, `riepilogo` |
| `frammento_1d.py` | `Params`, `Domain` (periodico / Neumann), `G0` / `GX` / `GY`, `History`, `FrammentoDelVeloce` (`project_novelty`, `project_mass`, `fidelity_test`, `report`), `plot` |
| `frammento_gf.py` | `verifica_quiete` (err_fx_fo, err_fo_ess, novita_rel, fraz dV<=0), `certifica_frammento` (quiete AND qualita' AND budget, mai se quiete=False), `correggi_micro_errori` (proposta non certificante), `MemoriaGF` (chiave sha256 valori+shape+dx, salva/richiama, stats hit/miss), `diagnostica_gf` |
| `confronto_bio.py` | coerenza interna LFP + ponte reale onesto: `spettro_potenza`, `potenza_relativa_theta_gamma` (theta 4-8, gamma 30-60), `filtro_banda`, `indice_pac_theta_gamma` (MI Tort), `pac_vs_surrogato` (z vs ampiezza mescolata), `similarita_spettrale` (coseno), `valida_sistema_sintetico` (ok_interno, mai bio), `confronta_sintetico_vs_reale` (`validazione_biologica=False` sempre), `carica_eeg_csv`,
  `trend_carico` (Spearman pooled) + `p_permutazione_trend` (Fase 12) |
| `stress_500.py` | stress test domande g0->gf: `genera_domande` (bump casuali + repliche ogni 25), `interroga` (catena massa/qualita'/novita'/quiete/cert/cache condivisa), `esegui` (CSV + md + 2 pannelli in `output/output_test/`) |
| `rete_frammento.py` | rete che non distrugge: `ReteFrammento` (nucleo frozen + slot isolati + scrittura solo via gf + espandi/rifiuta), `ReteIngenuaCondivisa` (baseline P condiviso che deriva), `genera_cue` (cue indipendenti senza repliche), `esegui_test_1000` (certifica n_cert, impara n_nuove, ri-testa), `salva_report_r1000` (CSV + md + fig11); richiamo associativo: `cue_parziale` (blocco/casuale + rumore), `ricostruisci_associativo` (residuo/gx), `esegui_test_associativo` (shift di classe A->B), `salva_report_assoc` (CSV + md + fig13) |
| `fase15.py` | prove referee (Fase 15): `ReteReplay` (rehearsal con buffer FIFO), `ReteEWC` (EWC-lite in forma chiusa sul campo, analogo concettuale con decadimento online), `esegui_confronto_cl` (stessa sequenza cue, con `shift` di classe) + `sweep_pareto_cl`/`salva_report_cl` (CSV + md + fig15); `misura_plasticita`/`salva_report_plasticita` (tasso cert, Q, ms/ricordo, memoria, fig16); retrieval OOD: `seleziona_prior` (MSE sul visibile) + `test_ood_mix_retrieval`/`test_ood_rumore_retrieval`/`salva_report_ood` (CSV + md + fig17) |
| `fase16.py` | Fase 16 (sonno + dimenticare + transfer + repair): `sonno`/`distilla_gist` (gist = media Fx, slot intatti) + `valuta_sonno_mix`/`rumore`/`salva_report_sonno` (fig18); `evici_slot` (eta/q/uso) + `esegui_eviction_study`/`salva_report_eviction` (fig19); `misura_transfer` (BWT/FWT a 2 task) + `salva_report_transfer` (fig20); `seleziona_prior_validato` (fit->val) + `seleziona_prior_coarse_to_fine` + `valuta_retrieval_repair`/`salva_report_retrieval` (fig21) |
| `demo_figure.py` | `fig_tre_livelli`, `fig_evoluzione`, `fig_diagnostica`, `fig_metriche`, `fig_turing`, `fig_invariante`, `fig_lfp` |
| `studi.py` | `valuta`, `valuta_multiseed` (media ± std), `tempo_recupero` (twin experiment), `config_sweep`, `config_ablazione`, `main_sweep`, `main_sweep_multiseed`, `main_ablazione`, `main_ablazione_multiseed`, `fig_ablazione` |

---

## 2. Modello: concettuale, matematico, numerico

### 2.0 I tre piani (da non confondere)

1. **Modello concettuale** — l'idea: ogni ricordo è un campo che diffonde in
   uno spazio astratto delle rappresentazioni, articolato in tre livelli:
   essenza immutabile (g0), operativita' adattiva (gx), novita' creativa
   vincolata (gy).
2. **Modello matematico** — le equazioni di reazione-diffusione (§2.1), i
   vincoli di massa (§2.3) e la forma normalizzata dell'invarianza (§2.4).
3. **Implementazione numerica** — Eulero esplicito / Eulero-Maruyama su
   griglia periodica, passo stabile CFL, rumore bianco o OU, test automatici
   (§6 Stabilita' numerica, §3 Test).

### 2.1 Equazioni 2D (modello principale, SDE di Ito')

```text
dF0 = D0 lap(F0) dt                                     (R_g0 = 0)
dFx = [Dx lap(Fx) + alpha (Fin - Fx) + kx (F0 - Fx)] dt
dFy = [Dy lap(Fy) + beta Fy (1 - Fy/K) G - ky Fy] dt + gamma G dW
```

con `dW = xi*sqrt(dt)`, schema di Eulero-Maruyama (parte deterministica =
Eulero esplicito). `Fy` e' proiettata su `>= 0` dopo ogni passo (densita'
non negativa, come nel 1D: rettifica il rumore e rende la novita' media
sensibile a `gamma`).

- `G` = gate di compatibilita': `clip(Fx / (1.2 max(Fx)), 0, 1) * mask(g0)`.
- `mask(g0)` = supporto strutturale dell'essenza (`essenza > 1e-3`).
- `xi` = rumore bianco `N(0,1)` oppure OU colorata a **varianza stazionaria
  unitaria** (`d(ou) = -ou/tau dt + sqrt(2/tau) dW`, `tau = 0.05`).
- **Condizioni al contorno**: toro `[0, L)^2`, laplaciano periodico con
  `np.roll` (varieta' compatta, senza bordo).
- **Condizioni iniziali**: `F0 = Fx = essenza` (due tracce mnestiche
  gaussiane), `Fy` = piccolo germe casuale sul supporto di g0.

### 2.2 Stabilita': funzione di Lyapunov pesata

Nel modello numerico la stabilita' viene valutata mediante una funzione di
Lyapunov pesata che misura la distanza dall'equilibrio omogeneo di g0, la
deviazione del livello gx dall'essenza e l'energia residua della novita' gy:

```text
V = w0 ||F0 - F0*||^2 + wx ||Fx - F0||^2 + wy ||Fy||^2
```

con `F0*` = equilibrio omogeneo della diffusione pura di g0 (non, in
generale, un equilibrio dell'intero sistema accoppiato). Nei run standard
considerati, la quantita' `V(t)` e' risultata non crescente in tutti i passi
temporali analizzati (frazione `dV/dt <= 0` al 100%). **Questa e' una verifica
numerica, non ancora una dimostrazione matematica generale.**

### 2.3 Conservazione della massa

Per g0 la conservazione e' esatta: il laplaciano periodico ha integrale
nullo, quindi `d/dt ∫F0 = 0` a meno dell'errore di arrotondamento (verificato:
`1.000000 → 1.000000`). Per i livelli adattivo (gx) e creativo (gy) i termini
di reazione possono modificare la massa: essa viene imposta mediante un
vincolo di budget e una proiezione numerica nello spazio degli stati
ammissibili,

```text
Pi_M(F) = F * M0 / ∫F dx
```

gia' implementata nel simulatore 1D (`project_mass`, `project_novelty` con
`novelty_budget`).

### 2.4 Stimatore della diffusivita' (lineare, non un invariante geometrico)

La forma `n = v` resta un **principio qualitativo di bilanciamento** (n e'
una massa, v una velocita'/diffusivita': hanno dimensioni diverse). La vecchia
forma `n = lambda_g * v_tilde` con `lambda_g = 1/v_tilde` per livello e'
**tautologica** (definisce `lambda_g` invece di misurarla) e non va usata
come prova — e non lo e' nemmeno quanto segue.

Definiamo, solo come controllo numerico dello schema:

```text
v = D_eff misurato da `<r^2> = 4 D t`,  v_tilde = v / D_g0
lambda_g = D_vero / v,    eta_g = v / D_vero = 1/lambda_g
```

Risultato (`N=48`, `python -m src 2d`):

```text
g0: v_tilde = 0.7934  lambda_g = 1.2604
gx: v_tilde = 0.1744  lambda_g = 1.1467
gy: v_tilde = 0.0356  lambda_g = 1.1236
media = 1.177 ± 0.060 (CV = 5.1%, D in [0.0020, 0.0500])
```

Lettura onesta: lo **stimatore della diffusivita' e' risultato lineare e
quasi non distorto su un range di D di 25x** (CV 5–7%). E' una verifica
numerica rispettabile della discretizzazione a `dx` fisso — non la prova di
un invariante geometrico della memoria: e' una proprieta' dello schema, e
`lambda_g` puo' spostarsi cambiando `dx`. Funzione: `verifica_invarianza(p)`.

### 2.5 Novita' controllata (Turing in gy)

Attivatore-inibitore di Gierer-Meinhardt confinato nel supporto di `g0`:

```text
ra = rho a^2 / (h + eps) - mu_a a
rh = rho a^2 - mu_h h
```

### 2.6 Metriche

- **Qualita'**: `1 - ||Fx - essenza|| / ||essenza||`.
- **Continuita'**: coseno tra frame consecutivi di `gx`.
- **Fedelta'**: correlazione tra ricordo deterministico e stocastico. Negli
  studi di robustezza e' calcolata su **gy** (il rumore entra solo li': su gx
  sarebbe identicamente 1).
- **Adattamento**: `1 - ||Fx - Fin|| / ||essenza - Fin||` (0 = resta
  sull'essenza, 1 = replica l'input; per `rilassamento` `Fin = 0`). La vecchia
  `corr(Fx, Fin)` era degenere (~-0.04 ovunque) perche' `Fx` e' dominata
  dall'essenza a due picchi mentre `Fin` e' un bump altrove.
- **Tempo di recupero**: twin experiment — gemello imperturbato vs gemello
  perturbato (bump gaussiano) in rilassamento deterministico; primo istante
  in cui l'energia della perturbazione scende sotto il 5% (`stato_iniziale`
  di `simula`). Finestra `T_rec = 1.00` (scala `~3/(alpha+kappa_x)`; con
  `T_rec = 0.30` quasi nulla recuperava per costruzione). L'attraversamento
  e' interpolato linearmente tra snapshot (snapshot fini, `salva_ogni=2`),
  non il primo punto di griglia.
- **LFP sintetico**: theta 6 Hz + gamma 45 Hz con ampiezza gamma modulata
  dalla novita' (analogia computazionale, vedi §7).

### 2.7 Livello gf (quiete attiva + certificazione + memoria)

Strato di garanzia sullo stato finale (`Fo`, `Fx`, `Fy`, `essenza`), senza
modificare la dinamica 2D. Solo `numpy` + `hashlib`.

```text
err_fo_ess  = ||Fo - essenza|| / (||essenza|| + eps)
err_fx_fo   = ||Fx - Fo|| / (||Fo|| + eps)
novita_rel  = ||Fy|| / (||essenza|| + eps)
quiete      = (err_fo_ess < delta) AND (err_fx_fo < eps)
              AND (novita_rel <= budget) AND (fraz dV<=0 >= 0.9 se V_hist >= 2 punti)
certificato = quiete AND (qualita' >= soglia) AND (novita_rel <= budget)
Fx_corr     = (1-f) Fx + f Fo   (proposta leggera, NON certifica)
```

- Default tarati (Fo diffonde con `D0=0.05`, quindi `||Fo-essenza|| ~0.66`
  su base `N=48 T=0.15` anche in regime sano): `eps=0.60`, `delta=0.90`,
  `soglia_qualita=0.40`, `budget_novita_rel=0.30` (0.25 in `verifica_quiete`).
- Anti-tautologia: niente auto-certificazione (la correzione va rivalutata
  con `verifica_quiete`/`certifica_frammento`), fedelta' solo su gy (non gx),
  soglie frozen, invalidazione cache su cambio valori/shape/`dx` (chiave
  sha256 di valori arrotondati a 1e-6 + shape + `dx`).
- `MemoriaGF`: `chiave` / `salva` / `richiama` → `(hit, payload)` /
  `stats` (`hits`, `misses`, `salvataggi`, `elementi`). Il recall non
  ricalcola nulla (costo ~0).
- Riferimento (`N=48, T=0.15`, `python -m src gf`): quiete SI
  (`err_fx_fo=0.4231<0.60`, `err_fo_ess=0.6636<0.90`), `Q=0.4877`,
  certificato SI, cache `hit1=True hit2=True`.

### 2.8 Confronto bio: coerenza interna LFP/theta-gamma (non validazione)

`lfp_sintetico` impone per costruzione theta 6 Hz + gamma 45 Hz con gamma
gated da theta e modulata dalla novita' — i picchi di fig07 sono quindi
tautologici (vedi §7). `src/confronto_bio.py` verifica solo la coerenza
interna e offre un confronto spettrale onesto con un tracciato reale:

- **Spettro**: periodogramma rFFT con Hann; potenze theta 4-8 Hz e gamma
  30-60 Hz + rapporto gamma/theta.
- **PAC**: Modulation Index di Tort (fase theta vs ampiezza gamma, 18 bin,
  MI in [0,1]) + `pac_vs_surrogato` (z vs ampiezza mescolata, 20 surrogati).
- **`valida_sistema_sintetico`** (bassa vs alta novita'): picchi 6/45 Hz
  presenti + potenza gamma che cresce con la novita' + PAC z>2. Riferimento
  (`N=16`, `fs=1000`, `durata=2.0`): novita' 1.0, rel_theta 0.70,
  rel_gamma 0.272, PAC MI 0.0915 vs surr 0.0007, **z=358** (atteso: gating
  imposto). Flag `validazione_biologica=False` sempre.
- **`confronta_sintetico_vs_reale`**: similarita' coseno tra spettri +
  potenze relative; motivo con `NON validazione biologica` esplicito.
- **`carica_eeg_csv`**: carica un canale da CSV (per dataset aperti
  pre-esportati: OpenNeuro/PhysioNet/TUH); nessun download automatico.

---

## 3. Requisiti

- Python 3.10+ (testato su 3.13)
- `numpy`, `matplotlib`, `scipy` (scipy usata da `confronto_bio`: Hilbert/PAC)

```bash
pip install -r requirements.txt
```

Nessuna compilazione richiesta. Le figure usano backend `Agg` (nessun display necessario).

### Con Docker (consigliato per riproducibilita')

```bash
# build
docker build -t frammento-del-veloce:latest .

# tutto in sequenza (2D + 1D + 7 figure, PNG in ./output)
docker compose run --rm frammento all
docker compose run --rm frammento demo

# comandi singoli
docker run --rm frammento-del-veloce:latest 2d --N 96 --T 0.30 --protocollo stimolo
docker run --rm -v ./output:/app/output frammento-del-veloce:latest demo
docker run --rm -v ./output:/app/output frammento-del-veloce:latest 1d --N1d 256 --T1d 2.0
```

Il `docker-compose.yml` monta `./output:/app/output`, quindi le figure
generate nel container restano disponibili sull'host.

### Test

```bash
# veloci di default (90 test, ~15 s; gli slow vengono skippati)
python -m pytest tests/ -q

# tutti, inclusi slow: demo + sweep/ablazione mini + rete 1000 + assoc (~6 min)
python -m pytest tests/ -q --run-slow

# solo gli slow
python -m pytest tests/ -q --run-slow -m slow

# solo il livello gf (7 test, <2 s, N=16, nessun file)
python -m pytest tests/test_gf.py -q

# solo coerenza LFP/PAC/confronto (8 test, ~7 s, nessun download)
python -m pytest tests/test_confronto_bio.py -q

# solo catena stress g0->gf (3 test, ~2 s, sottoinsieme N=16)
python -m pytest tests/test_stress_500.py -q

# solo rete che non distrugge (7 fast, ~6 s, N=16, nessun file)
python -m pytest tests/test_rete_1000.py -q

# solo richiamo associativo (7 fast, ~3 s, N=16, nessun file)
python -m pytest tests/test_rete_assoc.py -q

# solo Fase 15 CL/plasticita'/OOD (11 fast, ~8 s, N=16, report in tmp dir)
python -m pytest tests/test_fase15.py -q

# solo Fase 16 sonno/eviction/transfer/repair (12 fast, ~8 s, N=16, report in tmp dir)
python -m pytest tests/test_fase16.py -q

# con coverage (XML in output/coverage.xml)
python -m pytest tests/ -q --run-slow --cov=src --cov-report=term-missing

# dentro Docker
docker run --rm --entrypoint python frammento-del-veloce:latest -m pytest tests/ -q --run-slow
```

Copertura: conservazione massa g0, decrescita di Lyapunov, stima di `D`
entro un fattore 2 (forma normalizzata `v_tilde`), range di qualita'/
continuita', fedelta' su gy, twin di recupero, LFP, Turing vincolato a `g0`,
conservazione massa 1D, `fidelity_test`, CLI 2d/1d, sweep/ablazione, import demo,
quiete SI/NO, Fy esplosa, anti-tautologia (mai certificato se non quiete),
cache hit a costo zero, correzione non certificante, picchi 6/45 Hz imposti,
gamma che cresce con novita', PAC>surrogati, sim-sim>sim-rumore, confronto
onesto (`validazione_biologica=False`), CSV temp, catena stress g0->gf +
replica cache, trend/permutazione Fase 12, nucleo frozen + solo-gf-scrive +
capacita' espandi/rifiuta + interferenza zero + ingenua che degrada +
cache hit rete, cue parziali (blocco/casuale/rumore) + ricostruzione
deterministica + protetta invariante + condivisa accoppiata al set,
CL con shift (protetta 0 vs ingenua che degrada, EWC lam=0 = ingenua,
EWC tarato che riduce, replay con buffer grande meglio del FIFO corto),
plasticita' lineare + rifiuta/copertura, retrieval esatto in-distribution
e fragile al rumore + mix che recupera il lato giusto agli estremi,
sonno che non tocca gli slot + eviction misurata + BWT/FWT + retrieval
in validazione fit->val.
Totale **108 test**
(102 fast + 6 slow), coverage **83%** sul full run
(`confronto_bio.py` 90%, `studi.py` 97%, `frammento_2d.py` 94%,
`frammento_gf.py` 73%, `rete_frammento.py` 69%, `fase15.py` 85%,
`fase16.py` 97%,
`stress_500.py` 34% — gli script full girano fuori CI).

---

## 4. Avvio rapido

```bash
# tutto in sequenza: simulazione 2D + simulatore 1D + 7 figure
python run.py all

# oppure via modulo
python -m src all
```

Output atteso (valori di riferimento su `N=96, T=0.30`):

```text
--- FRAMMENTO DEL VELOCE : diagnostica numerica ---
griglia            : 96 x 96, dt = 2.17e-04
massa g0  iniziale : 1.000000 | finale: 1.000000
massa gx  iniziale : 0.999364 | finale: 0.468873
novita' gy (fine)  : 0.011574
gate gx medio      : 0.0935
Lyapunov V(0)      : 4.157018 | V(fine): 0.401803
frazione dV/dt<=0  : 100.0 %
g0: n = 1.0000  v = 0.037711  v~ = 0.7542  D_stimato = 0.037602 (vero 0.050000)  lambda_g = 1.3259
gx: n = 1.0000  v = 0.008630  v~ = 0.1726  D_stimato = 0.008629 (vero 0.010000)  lambda_g = 1.1588
gy: n = 1.0000  v = 0.001781  v~ = 0.0356  D_stimato = 0.001781 (vero 0.002000)  lambda_g = 1.1231
invarianza: lambda_g g0/gx/gy = 1.326 / 1.159 / 1.123  media = 1.203 ± 0.088 (CV = 7.4%, D in [0.0020, 0.0500])
```

---

## 5. Uso dettagliato

### 5.1 Simulazione 2D

```bash
python -m src 2d --N 96 --T 0.30 --protocollo stimolo
python -m src 2d --N 96 --T 0.30 --protocollo rilassamento
```

Uso da codice:

```python
from src.frammento_2d import Param, simula, riepilogo, invariante_nv, fedelta

p = Param(N=96)
snap = simula(p, T=0.30, protocollo="stimolo")
print(riepilogo(snap))

for nome, D in (("g0", p.D0), ("gx", p.Dx), ("gy", p.Dy)):
    print(nome, invariante_nv(p, D))

# fedelta' con / senza rumore
snap_a = simula(p, T=0.30, stocastico=False)
snap_b = simula(p, T=0.30, stocastico=True)
print("fedelta':", fedelta(snap_a, snap_b))
```

Parametri principali di `Param`:

| Parametro | Default | Significato |
|---|---|---|
| `N`, `L` | 96, 1.0 | griglia `N x N` su toro di lato `L` |
| `D0`, `Dx`, `Dy` | 0.05, 0.01, 0.002 | diffusioni di g0 / gx / gy |
| `alpha`, `kappa_x` | 3.0, 0.6 | accoppiamento gx-input / richiamo a g0 |
| `beta`, `K` | 0.9, 1.2 | crescita logistica / capacita' di novita' |
| `gamma`, `kappa_y` | 0.02, 0.25 | rumore creativo / smorzamento gy |
| `soglia` | 0.02 | (riservata a varianti con gate a soglia) |
| `seed` | 7 | riproducibilita' RNG |

Opzioni di `simula`:

| Argomento | Default | Note |
|---|---|---|
| `T`, `dt` | 0.30, `dt_stabile()` | `dt <= dx^2 / (4 Dmax)` per Eulero esplicito |
| `stocastico` | True | False = versione deterministica |
| `protocollo` | `"stimolo"` | `"stimolo"` (Lissajous) oppure `"rilassamento"` (input nullo) |
| `salva_ogni` | 50 | passo di snapshot |
| `rumore_bianco` | True | False = rumore OU colorato |
| `stato_iniziale` | None | `{'F0','Fx','Fy'}`: riparte da uno stato dato (twin di recupero) |

### 5.2 Simulatore 1D

```bash
python -m src 1d --N1d 256 --T1d 2.0
python -m src 1d --N1d 128 --T1d 0.5 --plot output/prova_1d.png
python -m src 1d --no-mass --evolve-g0 --gamma 0.05 --beta 1.4 --budget 0.3
```

Uso da codice:

```python
from src.frammento_1d import Params, FrammentoDelVeloce, plot

p = Params(T=2.0, N=256, seed=0)
sim = FrammentoDelVeloce(p)
sim.run()
print("fedelta':", sim.fidelity_test())
plot(sim, "output/frammento_1d.png")
```

### 5.3 Demo e figure

```bash
python -m src demo
```

Produce in `output/`:

| File | Contenuto |
|---|---|
| `fig01_tre_livelli.png` | campi g0, input `x(t)`, gx, gy a `t` finale |
| `fig02_evoluzione.png` | snapshot temporali di gx e gy |
| `fig03_lyapunov_massa.png` | `V(t)`, masse, `dV/dt` |
| `fig04_metriche.png` | qualita' / continuita' |
| `fig05_turing_gy.png` | pattern attivatore-inibitore vincolati da g0 |
| `fig06_invariante_nv.png` | `⟨r²⟩ = 4Dt` + barre `v/D_g0`, `D_stim/D` |
| `fig07_lfp.png` | traccia LFP sintetica + spettro theta/gamma |
| `fig08_ablazione.png` | barre qualita'/continuita'/fedelta'/adattamento per baseline (da `ablazione`) |

La demo usa una configurazione dimostrativa (`N=80`, `T=0.22`) definita in
`src/demo_figure.py:main()` per tempi di calcolo contenuti.

### 5.4 Studi di robustezza (sweep + ablazione)

```bash
python -m src sweep --N 48 --T 0.15        # 7 configurazioni -> studio_parametri.csv/.md
python -m src ablazione --N 48 --T 0.15    # 6 baseline -> ablazione.csv/.md + fig08
# multi-seed con media ± std (consigliato per la significativita'):
python -m src sweep --N 48 --T 0.15 --seeds 7 11 13 21 33
python -m src ablazione --N 48 --T 0.15 --seeds 7 11 13 21 33
```

Vedi §6 per le tabelle dei risultati.

### 5.5 Livello gf (quiete + certificazione + memoria)

```bash
python -m src gf --N 48 --T 0.15
python -m src gf --N 48 --T 0.15 --protocollo rilassamento
```

Output atteso (base `N=48, T=0.15`):

```text
GF quiete=SI (err_fx_fo=0.4231, err_fo_ess=0.6636, nov_rel=0.0056) | cert=SI (Q=0.488, ad=n/d, nov_rel=0.0056) | certificato: quiete attiva, qualita=0.4877>=0.4, novita_rel=0.0056<=0.3
motivo: certificato: quiete attiva, qualita=0.4877>=0.4, novita_rel=0.0056<=0.3
cache: chiave 39d2a41dcbb9... salva->richiama hit1=True hit2=True (costo ricomputazione ~0) stats={'hits': 2, 'misses': 0, 'salvataggi': 1, 'elementi': 1}
```

Uso da codice:

```python
from src.frammento_2d import Param, simula
from src.frammento_gf import verifica_quiete, certifica_frammento, MemoriaGF, correggi_micro_errori

p = Param(N=48, seed=7)
snap = simula(p, T=0.15, protocollo="stimolo", salva_ogni=20)
Fo, Fx, Fy, ess = snap["F0"][-1], snap["Fx"][-1], snap["Fy"][-1], snap["essenza"]

q = verifica_quiete(Fo, Fx, ess, p.dx, Fy=Fy, V_hist=snap["diag"]["V"])
c = certifica_frammento(Fo, Fx, Fy, ess, p.dx, diag=snap["diag"])
print(c["motivo"])  # certificato solo se quiete attiva

# micro-correzione (non certifica: va rivalutata)
Fx_corr = correggi_micro_errori(Fx, Fo, fattore=0.1)

# memoria a costo zero (solo se certificato)
mem = MemoriaGF()
k = mem.chiave(Fo, Fx, Fy, ess, dx=p.dx)
if c["certificato"]:
    mem.salva(k, {"qualita": c["qualita"], "quiete": q})
    hit, payload = mem.richiama(k)  # nessun ricalcolo
```

### 5.6 Confronto bio (coerenza interna + ponte reale)

```python
from src.frammento_2d import Param, essenza, lfp_sintetico
from src.confronto_bio import valida_sistema_sintetico, confronta_sintetico_vs_reale, carica_eeg_csv

p = Param(N=16, seed=7)
ess = essenza(p)
s_bassa = lfp_sintetico(p, ess * 0 + ess.mean(), fs=1000.0, durata=1.0, seed=3)
s_alta = lfp_sintetico(p, ess, fs=1000.0, durata=1.0, seed=3)
print(valida_sistema_sintetico(s_bassa["lfp"], s_alta["lfp"], fs=1000.0)["motivo"])

# con un tracciato reale pre-esportato in CSV (una colonna = un canale)
reale = carica_eeg_csv("dati/eeg_canale.csv", colonna=0, fs=256.0)
print(confronta_sintetico_vs_reale(s_alta["lfp"][:reale["n_campioni"]], reale["segnale"], fs=256.0)["motivo"])
```

8 test in `tests/test_confronto_bio.py` (nessun download, <10 s).

### 5.7 Stress test 500 domande (g0->gf)

```bash
python -m src stress500 --n 500 --N 32 --T 0.10 --seed 7
```

Ogni domanda = cue di richiamo (bump gaussiano casuale + seed/protocollo/
gamma casuali); risposta = catena massa g0 → qualita' gx → novita' gy →
quiete/cert/cache gf con `MemoriaGF` condivisa (ogni 25 domande una replica
esatta per provare il recall). Output in `output/output_test/` (versionato
su GitHub): `stress_500.csv` (500 righe), `stress_500.md`,
`fig_stress_pannello1.png` (tassi + istogrammi), `fig_stress_pannello2.png`
(scatter eq/ea + qualita' nel tempo + hit cumulati). Senza ricalcolare:
script `Temp/opencode/rigenera_stress.py` rigenera le figure dal CSV.

### 5.8 Rete che non distrugge — test dei 1000 (interferenza retroattiva)

```bash
python -m src rete1000 --n-cert 200 --n-nuove 800 --N 32 --T 0.10 --seed 7
```

Tesi: imparare cose nuove non distrugge mai quelle certificate (nucleo g0
frozen con checksum, slot isolati, scrittura solo via gf, capacita'
espandi/rifiuta). Protocollo: certifica 200 cue, impara 800 sopra, ri-testa
le 200 → zero degradi o fallimento. Output in `output/output_test/`
(versionato): `rete_1000.csv` (200 righe prima/dopo + baseline),
`rete_1000.md`, `fig11_rete_Q.png` (Q piatta + scatter prima/dopo).

Uso da codice:

```python
from src.rete_frammento import ReteFrammento, esegui_test_1000
res = esegui_test_1000(n_cert=200, n_nuove=800, N=32, T=0.10, seed=7)
print(res["max_degrado"], res["distrutti"], res["nucleo_ok"])  # 0.0 0 True
```

### 5.9 Richiamo associativo (cue parziali → ricostruzione) + shift di classe

```bash
python -m src assoc --n-cert 200 --n-nuove 800 --N 32 --T 0.10 --seed 7
```

Richiamo da cue **parziali** (non ri-esecuzione): il cue e' una versione
parziale di `Fx` certificato (blocco centrale o pixel sparsi + rumore);
la ricostruzione usa la dinamica condivisa del modello — `residuo`:
`Fx_rec = Fo + R`, con `R = Fx - Fo` esteso armonicamente (diffusione pura,
ancoraggio sui pixel osservati); alternativa `gx` (rilassamento verso il
prior). Set A (200, bump a sinistra) poi set B (800, bump a destra): shift
di classe. Output in `output/output_test/`: `assoc_1000.csv` (600 righe =
200 ricordi × 3 frazioni), `assoc_1000.md`, `fig13_assoc.png`.

Uso da codice:

```python
from src.rete_frammento import esegui_test_associativo, salva_report_assoc
res = esegui_test_associativo(n_cert=200, n_nuove=800, N=32, T=0.10, seed=7)
print(res["verifica_protetta_max_diff"], res["agg"][0.75]["degrado_ing"])
```

### 5.10 Prove referee — Fase 15 (CL con shift + plasticita' + OOD)

```bash
# Pareto CL + plasticita' fino a 800 + OOD con retrieval (N=16, ~1 min)
python -m src fase15 --n-cert 30 --n-nuove 60 --N 16 --T 0.05 --seed 7
```

Stessa sequenza di cue per 4 modelli (protetta / ingenua / replay / EWC-lite)
con shift di classe A->B; misura di plasticita' (tasso certificazione,
Q media, ms/ricordo, memoria); retrieval del prior sulla parte visibile
(MSE) + mix composizionali e sweep di rumore. Output in
`output/output_test/`: `fase15_cl_pareto.csv/.md` + `fig15_cl_pareto.png`,
`fase15_plasticita.csv/.md` + `fig16_plasticita.png`,
`fase15_ood.csv/.md` + `fig17_ood.png`.

Uso da codice:

```python
from src.fase15 import sweep_pareto_cl, misura_plasticita
from src.fase15 import test_ood_mix_retrieval, test_ood_rumore_retrieval
righe = sweep_pareto_cl(n_cert=30, n_nuove=60, N=16, T=0.05, seed=7, shift=True)
print(righe[0])
```

### 5.11 Fase 16 — sonno + dimenticare + transfer + repair

```bash
# Sonno (gist) + eviction + BWT/FWT + retrieval repair (N=16, ~1 min)
python -m src fase16 --n-cert 30 --n-nuove 60 --N 16 --T 0.05 --seed 7
```

Sonno offline (`sonno`/`distilla_gist`: gist = media degli Fx, verifica
hash slot + checksum nucleo prima/dopo); eviction esplicita
(`evici_slot` eta/q/uso, mai sovrascrittura); transfer standard a 2 task
(`misura_transfer`: BWT/FWT); retrieval repair (`seleziona_prior_validato`
fit->val + `seleziona_prior_coarse_to_fine`). Output in
`output/output_test/`: `fase16_sonno.csv/.md` + `fig18_sonno.png`,
`fase16_eviction.csv/.md` + `fig19_eviction.png`,
`fase16_transfer.csv/.md` + `fig20_transfer.png`,
`fase16_retrieval.csv/.md` + `fig21_retrieval.png`.

Uso da codice:

```python
from src.fase16 import sonno, esegui_eviction_study, misura_transfer
from src.fase16 import valuta_retrieval_repair
s = sonno(rete)  # {"gist": ..., "intatto": True, ...}
print(s["intatto"], s["dettagli"]["std_pixel"])
```

---

## 6. Studi di robustezza

Condizioni: `N=48`, `T=0.15`, `T_rec=1.00`. `t_rec` con `+` = non recuperato
entro la finestra (limite inferiore onesto). CSV completi in `output/`:
`studio_parametri.csv`, `studio_parametri_multiseed.csv`,
`ablazione.csv`, `ablazione_multiseed.csv`.

Nota di disegno: qualita'/continuita'/`max dV/dt` dipendono da g0/gx
(deterministici) e **devono** restare identici al variare di `gamma` — la
sonda del rumore e' la fedelta' di gy (det vs stoc) con Eulero-Maruyama
(`sqrt(dt)`) e la novita' media (rettifica `Fy >= 0`).
Nelle tabelle multi-seed le colonne con `± 0.0000` (qualita', adattamento,
`t_rec`) sono **indipendenti dal seed per costruzione** (sottosistema
g0/gx deterministico), non un errore di calcolo.

### 6.1 Sweep dei parametri (singolo seed 7)

| config | massa g0 | qualita' | continuita' | fedelta' (gy) | max dV/dt | dV/dt<=0 | novita' | t_rec |
|---|---|---|---|---|---|---|---|---|
| A base (stimolo, bianco) | 1.0000 | 0.4877 | 0.9995 | 0.9986 | -6.83 | 100% | 0.0121 | 0.487 |
| B rumore nullo | 1.0000 | 0.4877 | 0.9995 | 1.0000 | -6.83 | 100% | 0.0121 | 0.487 |
| C rumore alto (γ=0.15) | 1.0000 | 0.4877 | 0.9995 | 0.9508 | -6.83 | 100% | 0.0123 | 0.487 |
| D rumore OU colorato | 1.0000 | 0.4877 | 0.9995 | 0.9480 | -6.83 | 100% | 0.0121 | 0.487 |
| E creativita' alta (β=2.0) | 1.0000 | 0.4877 | 0.9995 | 0.9986 | -6.83 | 100% | 0.0123 | 0.487 |
| F rilassamento | 1.0000 | 0.4872 | 0.9995 | 0.9986 | -6.80 | 100% | 0.0121 | 0.487 |
| G adattamento forte (α=8.0) | 1.0000 | **0.2554** | 0.9994 | 0.9986 | -5.45 | 100% | 0.0121 | 0.239 |

### 6.2 Sweep multi-seed (5 seed: 7, 11, 13, 21, 33 — media ± std)

| config | qualita' | fedelta' (gy) | novita' | adattamento | t_rec |
|---|---|---|---|---|---|
| A base | 0.4877 ± 0.0000 | 0.9988 ± 0.0002 | 0.0119 ± 0.0002 | 0.4793 ± 0.0000 | 0.487 ± 0.000 |
| B rumore nullo | 0.4877 ± 0.0000 | 1.0000 ± 0.0000 | 0.0120 ± 0.0002 | 0.4793 ± 0.0000 | 0.487 ± 0.000 |
| C rumore alto (γ=0.15) | 0.4877 ± 0.0000 | 0.9493 ± 0.0057 | 0.0121 ± 0.0002 | 0.4793 ± 0.0000 | 0.487 ± 0.000 |
| D OU colorato | 0.4877 ± 0.0000 | 0.9502 ± 0.0066 | 0.0119 ± 0.0002 | 0.4793 ± 0.0000 | 0.487 ± 0.000 |
| E β=2.0 | 0.4877 ± 0.0000 | 0.9988 ± 0.0002 | 0.0121 ± 0.0002 | 0.4793 ± 0.0000 | 0.487 ± 0.000 |
| F rilassamento | 0.4872 ± 0.0000 | 0.9988 ± 0.0002 | 0.0119 ± 0.0002 | 0.4798 ± 0.0000 | 0.487 ± 0.000 |
| G α=8.0 | **0.2554** ± 0.0000 | 0.9988 ± 0.0002 | 0.0119 ± 0.0002 | **0.7312** ± 0.0000 | **0.239** ± 0.000 |

Lettura: il rumore ora ha effetto reale e significativo — A vs B vs C sulla
fedelta' (0.9988 ± 0.0002 vs 1.0000 vs 0.9493 ± 0.0057: differenze ≫ std);
β alza la novita' (E: 0.0121 vs A: 0.0119, significativo ma di effetto
contenuto); α forte dimezza la qualita' ma raddoppia l'adattamento (0.73 vs
0.48) e dimezza `t_rec`. Stabile ovunque (`dV/dt<=0` 100%, massa g0 esatta);
recupero al 100% con `T_rec=1.00`.
`t_rec` e' identico tra A–F per disegno (il twin perturba solo `Fx`, il cui
sottosistema deterministico non dipende da gy/rumore: stesse traiettorie,
stesso attraversamento interpolato); solo G (α diverso) si distingue.

### 6.3 Ablazione dei livelli (singolo seed 7)

| config | massa g0 | qualita' | novita' | fedelta' | dV/dt<=0 | t_rec | adattamento |
|---|---|---|---|---|---|---|---|
| 1 solo diffusione | 1.0000 | 0.7088 | 0.0119 | 1.0000 | 100% | 1.000+ | 0.2040 |
| 2 g0+gx (senza gy) | 1.0000 | 0.4877 | 0.0119 | 1.0000 | 100% | 0.487 | 0.4793 |
| 3 completo | 1.0000 | 0.4877 | 0.0121 | 0.9986 | 100% | 0.487 | 0.4793 |
| 4 senza vincolo (ky=0, K=5) | 1.0000 | 0.4877 | **0.0125** | 0.9987 | 100% | 0.487 | 0.4793 |
| 5 senza rumore | 1.0000 | 0.4877 | 0.0121 | 1.0000 | 100% | 0.487 | 0.4793 |
| 6 rumore eccessivo (γ=0.25) | 1.0000 | 0.4877 | **0.0128** | 0.8924 | 100% | 0.487 | 0.4793 |

### 6.4 Ablazione multi-seed (media ± std)

| config | qualita' | fedelta' | novita' | adattamento | t_rec |
|---|---|---|---|---|---|
| 1 solo diffusione | 0.7088 ± 0.0000 | 1.0000 ± 0.0000 | 0.0118 ± 0.0002 | 0.2040 ± 0.0000 | 1.000 (0% rec) |
| 2 g0+gx | 0.4877 ± 0.0000 | 1.0000 ± 0.0000 | 0.0118 ± 0.0002 | 0.4793 ± 0.0000 | 0.487 (100%) |
| 3 completo | 0.4877 ± 0.0000 | 0.9988 ± 0.0002 | 0.0119 ± 0.0002 | 0.4793 ± 0.0000 | 0.487 (100%) |
| 4 senza vincolo | 0.4877 ± 0.0000 | 0.9988 ± 0.0002 | 0.0124 ± 0.0002 | 0.4793 ± 0.0000 | 0.487 (100%) |
| 5 senza rumore | 0.4877 ± 0.0000 | 1.0000 ± 0.0000 | 0.0120 ± 0.0002 | 0.4793 ± 0.0000 | 0.487 (100%) |
| 6 γ=0.25 | 0.4877 ± 0.0000 | 0.8864 ± 0.0140 | 0.0125 ± 0.0002 | 0.4793 ± 0.0000 | 0.487 (100%) |

Lettura: g0 da solo = qualita' max (0.71) ma adattamento basso (0.20) e nessun
recupero senza accoppiamento (t_rec = finestra, 0%); gx porta l'adattamento a
0.48 al prezzo della somiglianza; gy aggiunge novita'; senza vincolo la
novita' cresce (+4%, significativo ma di effetto contenuto: il vincolo
trattiene); γ=0.25 crolla la fedelta' (0.89 ± 0.01) e alza la novita' (+5%,
significativo ma di effetto contenuto).

### 6.5 Livello gf (quiete + certificazione, N=48 T=0.15 det.)

| config | err_fx_fo (<0.60) | err_fo_ess (<0.90) | qualita' (>=0.40) | quiete | cert |
|---|---|---|---|---|---|
| A base (stimolo, bianco) | 0.4231 | 0.6636 | 0.4877 | SI | SI |
| C rumore alto (γ=0.15) | 0.4231 | 0.6636 | 0.4877 | SI | SI |
| F rilassamento | — | — | 0.4872 | SI | SI |
| G adattamento forte (α=8.0) | — | — | 0.2554 | SI | NO (qualita' < soglia) |
| γ=0.25 (certificazione completa gf+G3) | SI | SI | 0.4877 | SI | NO (fedelta' gy 0.8864±0.0140, rifiutata) |

Lettura: il rumore entra solo in gy stocastica, quindi A e C sono identici
su quiete/qualita' per disegno (la sonda del rumore resta la fedelta' di gy,
§6.1–6.2); G resta quieto ma non certificato per qualita' sotto soglia;
γ=0.25 resta quieto in gf puro ma viene rifiutato nella certificazione
completa (fedelta' gy crollata). Comando: `python -m src gf --N 48 --T 0.15`.

### 6.6 Confronto bio: coerenza interna (`N=16`, `fs=1000`, 8 test)

| check | risultato |
|---|---|
| picchi 6/45 Hz dove imposti | SI (sopra mediana banda 2-12 / 20-70 Hz) |
| gamma cresce con novita' | SI (bassa→alta novita', `p_gamma` cresce) |
| PAC Tort reale vs surrogati | MI 0.0915 vs surr 0.0007, **z=358** (>2) |
| sim-sim vs sim-rumore (coseno) | sim-sim > sim-rumore |
| `confronta_sintetico_vs_reale` | similarita' in [0,1], `validazione_biologica=False` sempre |

Lettura: il sistema sintetico e' coerente per costruzione (gating theta su
gamma + novita'→gamma). Non e' evidenza biologica: lo z altissimo conferma
il gating imposto, non il cervello. Test: `pytest tests/test_confronto_bio.py`.

### 6.6b Fase 12: gruppo 10 soggetti EEG reali (ds005095, Sternberg)

Pool frontale F3/Fz/F4, passa-alto 1 Hz + ICA, retention 1.5–3.5 s,
gamma/theta 30–48/4–8 Hz → mediana per carico e soggetto. Dati grezzi
~3.1 GB mai nel repo; matrice 10×5 in `output/output_test/fase12_mat.npy`,
report in `output/output_test/bio_fase12.md`.

Medie di gruppo per carico 3/6/9/12/15: **0.610, 0.551, 0.466, 0.486, 0.527**.
Spearman pooled **rho=-0.009, p=0.95**; Friedman p=0.08 n.s.; permutazione
entro-soggetto (2000) **p=0.89**. Variabilità inter-soggetto ~10x.
Verdetto: **nullo di gruppo** — nessun trend carico→gamma/theta. Pipeline
statistica testata in `tests/test_bio_fase12.py` (trend sintetico rilevato,
rumore n.s., matrice reale nulla).

### 6.7 Stress 500 domande (`N=32`, `T=0.10`, seed 7)

| misura | risultato |
|---|---|
| quiete SI | 500/500 (100%) |
| certificati | 500/500 (100%) |
| cache hit repliche | 19/19 (100%) |
| qualita' media | 0.595 |
| massa g0 drift max | ~1e-16 |

Lettura: a bump moderati e orizzonte breve il sistema regge senza crolli e
la memoria richiama tutte le repliche a costo ~0. Il 100% di certificati e'
atteso in questo regime — per vedere bocciature servono bump forti/α=8
(vedi §6.5). Dati e figure versionati in `output/output_test/`.

### 6.8 Rete che non distrugge — test dei 1000 (`N=32`, `T=0.10`, seed 7)

| misura | risultato |
|---|---|
| cue certificate (prime 200) | 200/200 (100%) |
| checksum nucleo prima→dopo | invariato, ok=True |
| PROTETTA max degrado Q | 0.00e+00 |
| PROTETTA distrutti | 0/200 (SUCCESSO: zero degradi) |
| INGENUA (P condiviso) max degrado | 0.4035 |
| INGENUA distrutti | 28/200 |

Lettura: dopo 800 nuovi apprendimenti le 200 Q certificate sono bit-identiche
(sim deterministica + slot isolati + nucleo frozen: interferenza strutturalmente
impossibile, non fortuna). La baseline ingenua con campo plastico condiviso
degrada davvero (28 distrutti): la protezione non e' vacua. Dati e figura
(`rete_1000.csv/.md`, `fig11_rete_Q.png`) versionati in `output/output_test/`.

### 6.9 Richiamo associativo (`N=32`, `T=0.10`, seed 7, bump 3-8)

Set A 200 certificati (bump a sinistra), poi set B 800 (bump a destra):
`||P_B - P_A||/||P_A|| = 0.177`. Protetta verificata bit-identica
prima/dopo: `max diff = 0.00e+00`.

Operatori di completamento (protetta, prior = core):

| frazione | residuo | media | core | armonica | gx |
|---|---|---|---|---|---|
| 0.25 | **0.7139 ± 0.0210** | 0.0571 | 0.6553 | 0.4729 | 0.5132 |
| 0.50 | 0.6491 ± 0.0221 | 0.0099 | 0.6531 | 0.0499 | 0.3318 |
| 0.75 | 0.6528 ± 0.0173 | 0.0027 | 0.6415 | 0.0062 | 0.3376 |

Degrado in regime associativo (protetta vs condivisa):

| frazione | protetta | condivisa P_A | condivisa dopo shift | degrado medio | degrado max | >0.05 |
|---|---|---|---|---|---|---|
| 0.25 | 0.7139 | 0.9472 | 0.9459 | 0.0013 | 0.0103 | 0% |
| 0.50 | 0.6491 | 0.9104 | 0.8865 | 0.0240 | 0.0507 | 1% |
| 0.75 | 0.6528 | 0.9008 | 0.8589 | 0.0420 | 0.0679 | 28% |

Lettura onesta: (i) la ricostruzione `residuo` batte i riempimenti semplici
e la diffusione senza nucleo (a f=0.25: +0.06 sul core, +0.24 sull'armonica
pura); (ii) **il prior condiviso batte il core per-ricordo in qualita'
assoluta** (0.86-0.95 vs 0.65-0.71): il pooling cattura la struttura comune
della classe, che domina questi campi — finding riportato cosi' com'e';
(iii) ma il richiamo condiviso e' **accoppiato al set**: dopo lo shift
degrada per il 93-100% dei ricordi vecchi, tanto piu' quanto piu' grande e'
il buco (fino a 0.068 singolo, 28% dei ricordi > 0.05 a f=0.75); la protetta
e' invariante per costruzione (0 esatto, verificato). Dati e figura
(`assoc_1000.csv/.md`, `fig13_assoc.png`) versionati in `output/output_test/`.

### 6.10 Pareto continual learning con shift (`N=16`, `T=0.05`, seed 7, A=30 sx + B=60 dx)

Stessa sequenza di cue, stessa metrica Q. Baseline ingenua (media mobile)
+ Replay (buffer FIFO) + EWC-lite (forma chiusa sul campo con decadimento
online `gamma=0.9`; `eta=0.1` come l'ingenua per confronto fair).

| modello | max degr | medio | distrutti | Q vecchie | Q nuove | costo extra |
|---|---|---|---|---|---|---|
| ingenua | 0.0462 | 0.0172 | 13/30 | 0.8350 | 0.8316 | 0 KB |
| replay K10 B30 (FIFO corto) | 0.0402 | 0.0136 | 12/30 | 0.8387 | 0.8287 | 60 KB |
| replay K10 B200 (tiene tutto) | 0.0312 | 0.0100 | 9/30 | 0.8422 | 0.8327 | 180 KB |
| replay K20 B200 | 0.0299 | 0.0085 | 8/30 | 0.8437 | 0.8346 | 180 KB |
| ewc lam=0.1 | 0.0239 | 0.0054 | 3/30 | 0.8468 | 0.8364 | 4 KB |
| ewc lam=0.2 | 0.0172 | 0.0026 | 0/30 | 0.8496 | 0.8392 | 4 KB |
| ewc lam=0.5 | 0.0104 | 0.0006 | 0/30 | 0.8516 | 0.8426 | 4 KB |
| ewc lam=1.0 | 0.0081 | 0.0001 | 0/30 | 0.8521 | 0.8447 | 4 KB |
| **protetta** | **0.0** | 0.0 | **0/30** | — | — | 4 campi/slot |

Lettura onesta: (i) senza shift tutti degradano poco (campi ~97% simili) —
lo shift e' necessario per separare i metodi; (ii) EWC-lite ben tarato
(`lam=0.2-1.0`, `eta=0.1`) arriva a **0 distrutti** con residuo max ~0.008
e impara B *meglio* dell'ingenua (Q nuove 0.8447 vs 0.8316) a 4 KB:
baseline forte, non strawman; (iii) Replay aiuta solo se il buffer trattiene
tutto (FIFO corto dimentica A dopo lo shift — stesso costo lineare degli
slot); (iv) la protetta vince sulla **garanzia strutturale** (0 esatto),
non sul margine: EWC si avvicina statisticamente ma senza garanzia.
`lam=0` coincide con l'ingenua per costruzione (sanity check nei test).
Dati e figura versionati (`fase15_cl_pareto.csv/.md`, `fig15_cl_pareto.png`).

### 6.11 Plasticita' (`N=16`, `T=0.05`, seed 7)

| n richieste | certificati | tasso | Q media | ms/ricordo | memoria |
|---|---|---|---|---|---|
| 50 | 50 | 1.000 | 0.9479 | 2.13 | 400 KB |
| 100 | 100 | 1.000 | 0.9483 | 1.95 | 800 KB |
| 200 | 200 | 1.000 | 0.9492 | 1.87 | 1600 KB |
| 400 | 400 | 1.000 | 0.9488 | 1.87 | 3200 KB |
| 800 | 800 | 1.000 | 0.9488 | 1.96 | 6400 KB |

Regime a memoria limitata (`rifiuta`, 800 richieste): cap 100 →
memorizzati 100 (copertura 0.125, Q 0.9483); cap 200 → 200 (0.250, Q 0.9492).
Lettura onesta ("gabbia dorata" quantificata): Q piatta, costo lineare
(~2 ms/ricordo, 8 KB/slot a N=16); il gate in regime facile non filtra
(tasso 1.0 anche con bump 3-8) — la saturazione emerge come rifiuto per
capacita', non come degrado. Dati e figura versionati
(`fase15_plasticita.csv/.md`, `fig16_plasticita.png`).

### 6.12 OOD con retrieval (`N=16`, `T=0.05`, seed 7)

Retrieval = MSE sulla parte visibile (mai oracolo), completamento `residuo`
col prior recuperato. Mix A+B: estremi perfetti (alpha=0→B margine 4e7,
alpha=1→A margine 1e5), centro ambiguo (alpha=0.5→B margine 0.16,
alpha=0.75→A margine 6e-4) ma `q_vs_mix` resta 0.83-0.88 ovunque.
Rumore: accuratezza retrieval 1.0 → 0.25 → 0.125 → 0.125 per
rumore 0 → 0.5 → 1.0 → 2.0.
Lettura onesta (risultato negativo pubblicabile): il retrieval e' exact-match
in-distribution e fragile al rumore; sui mix composizionali il sistema
**completa, non ragiona** — produce sempre un output plausibile col prior
recuperato. Dati e figura versionati (`fase15_ood.csv/.md`, `fig17_ood.png`).

### 6.13 Sonno (gist offline, `N=16`, `T=0.05`, seed 7, soglia margine 0.05)

Gist = media degli Fx certificati (8 slot: 4 sx + 4 dx), distillato in
lettura sola (hash slot + checksum nucleo identici prima/dopo, verificato).
Politica: se margine < 0.05 usa il gist, altrimenti il vincitore.

| alpha | vincitore | margine | q winner | q gist | q scelta |
|---|---|---|---|---|---|
| 0.00 | B | 4.36e+07 | 0.8324 | 0.8707 | 0.8324 |
| 0.25 | B | 2.14e-01 | 0.8623 | 0.9092 | 0.8623 |
| 0.50 | B | 1.59e-01 | 0.8775 | 0.9296 | 0.8775 |
| 0.75 | A | 6.53e-04 | 0.8738 | 0.9190 | 0.9190 |
| 1.00 | A | 1.44e+05 | 0.8542 | 0.8878 | 0.8542 |

Rumore (8 cue, q_mask media winner vs gist): 0.0 -> 0.8418 vs 0.8655
(acc 1.0); 0.5 -> 0.8105 vs 0.8262 (acc 0.25); 1.0 -> 0.7303 vs 0.7389
(acc 0.125); 2.0 -> 0.5262 vs 0.5288 (acc 0.125).
Lettura onesta: il gist alza q di ~+0.04 su mix e ~+0.01-0.02 col rumore,
ma e' pooling sfocato (media), non ragionamento — vince dove il mix stesso
e' una media. Non risolve l'exact-match (accuracy resta 0.125 al rumore
forte). Dati e figura versionati (`fase16_sonno.csv/.md`, `fig18_sonno.png`).

### 6.14 Dimenticare apposta (`n=40`, `N=16`, seed 7)

Eviction = cancellazione (i rimasti restano bit-identici, zero degrado).

| politica | k | rimasti | copertura | Q rimasti | memoria | risparmio |
|---|---|---|---|---|---|---|
| eta | 10/20/30 | 30/20/10 | 0.75/0.50/0.25 | 0.9469/0.9465/0.9457 | 240/160/80 KB | 80/160/240 KB |
| q (deboli prima) | 10/20/30 | 30/20/10 | 0.75/0.50/0.25 | 0.9503/0.9531/0.9559 | 240/160/80 KB | 80/160/240 KB |
| uso (proxy id%3) | 10/20/30 | 30/20/10 | 0.75/0.50/0.25 | 0.9465/0.9476/0.9476 | 240/160/80 KB | 80/160/240 KB |

Base senza eviction: 40 slot, Q 0.9472, 320 KB. La politica per-Q alza la
media dei rimasti (potatura dei deboli, +0.009 a k=30); eta/uso la lasciano
piatta. Dati e figura versionati (`fase16_eviction.csv/.md`,
`fig19_eviction.png`).

### 6.15 Transfer standard BWT/FWT (`N=16`, seed 7, A=30 sx + B=60 dx)

| modello | R_AA | R_AB | R_BB | R_B0 | BWT | FWT |
|---|---|---|---|---|---|---|
| protetta | 0.8522 | 0.8522 | 0.8494 | 0.8494 | 0.0000 | 0.0000 |
| ingenua | 0.8309 | 0.8350 | 0.8316 | 0.8335 | +0.0041 | -0.0019 |
| replay K10 B200 | 0.8357 | 0.8422 | 0.8327 | 0.8370 | +0.0065 | -0.0043 |
| ewc lam=0.5 | 0.8472 | 0.8516 | 0.8426 | 0.8447 | +0.0044 | -0.0020 |

Lettura onesta: in media-Q il BWT e' ~0 per tutti (nessun forgetting medio
rilevabile) — la metrica media nasconde i casi peggiori, dove la condivisa
distrugge davvero (13/30 in §6.10, max degr 0.046). La protetta e' l'unica
con BWT=FWT=0 esatti per costruzione (isolamento, non transfer). FWT ~0
ovunque: A non aiuta B. Dati e figura versionati
(`fase16_transfer.csv/.md`, `fig20_transfer.png`).

### 6.16 Retrieval repair (`N=16`, seed 7, k=3, val 20% del visibile)

| rumore | base | validato | coarse-to-fine |
|---|---|---|---|
| 0.00 | 1.000 | 0.250 | 0.250 |
| 0.50 | 0.250 | 0.250 | 0.250 |
| 1.00 | 0.125 | 0.250 | 0.250 |
| 2.00 | 0.125 | 0.250 | 0.250 |

Gara tra top-3 prior in validazione fit->val (mai l'occulto).
Lettura onesta (negativo con sfumatura): il repair raddoppia l'accuracy al
rumore forte (0.125 -> 0.250) ma crolla sul pulito (1.0 -> 0.25) — tenere da
parte il 20% del visibile toglie potere discriminante dove i campi sono
~97% simili. Coarse-to-fine identico al validato (la scrematura grossolana
non aggiunge). Il retrieval resta fragile. Dati e figura versionati
(`fase16_retrieval.csv/.md`, `fig21_retrieval.png`).

## 7. Limiti e natura del modello

- **Teorico-computazionale**, non validato su dati biologici (nessun
  EEG/fMRI/comportamentale reale a supporto). `src/confronto_bio.py` verifica
  solo la coerenza interna del sintetico + similarita' spettrale onesta
  (`validazione_biologica=False` sempre). Piloti esplorativi su EEG reale
  (OpenNeuro ds005095: N=1 e poi gruppo N=10, Sternberg, retention):
  nessun effetto carico→gamma/theta (Spearman rho=-0.009, p=0.95;
  permutazione p=0.89) — report locali in `output/output_test/bio_fase12.md`.
  Risultati nulli, non validazione.
- **Sensibile ai parametri**: oltre una soglia di adattamento (α≈8) la
  qualita' si dimezza; i vincoli di gy sono necessari (config 4, +4% novita'
  senza vincolo).
- **Spazio cognitivo astratto**: nessuna interpretazione neuroanatomica
  diretta di g0/gx/gy per ora.
- **Rumore**: schema di Eulero-Maruyama corretto (`gamma*G*dW`,
  `dW = xi*sqrt(dt)`, OU a varianza unitaria, `Fy >= 0`). La fedelta' di gy
  scende con `gamma` (1.000 → 0.999 → 0.95 → 0.89) e la novita' media cresce
  (+5% a γ=0.25 per rettifica); qualita'/`dV/dt` restano insensibili per
  disegno (dipendono da gx/g0 deterministici).
- **Recupero**: con `T_rec = 1.00` (scala `~3/(alpha+kappa_x)`) il twin
  recupera al 100% (0.49 base, 0.24 con α=8); senza accoppiamento
  (solo diffusione) non recupera entro la finestra.
- **Orizzonte breve**: sweep/ablazione a `T=0.15`; comportamenti su tempi
  lunghi e pattern di Turing completi restano da esplorare.
- **gf**: soglie tarate sul regime sano (`eps=0.60`, `delta=0.90`,
  `soglia_qualita=0.40`), non principi primi — con `Fo` che diffonde,
  soglie strette renderebbero la quiete impossibile per disegno. "Costo
  zero" = nessun ricalcolo al recall (la costruzione resta O(simula));
  la cache non riusa mai certificati obsoleti (chiave sha256 su
  valori+shape+`dx`). La correzione `Fx_corr` non certifica da sola.
- **rete 1000**: orizzonte breve (`T=0.10`), bump moderati (amp 0.5–2.0),
  sim deterministiche per isolare l'interferenza dal rumore; lo zero degrado
  e' strutturale (isolamento + frozen), non una misura di generalizzazione.
- **richiamo associativo**: la ricostruzione e' il limite di diffusione del
  modello (nessun apprendimento di un readout); il gate di certificazione
  (`eps=0.60`) limita quanto i ricordi possono differire dal nucleo, quindi
  limita anche l'interferenza misurabile (fino a ~0.07 per ricordo qui);
  il prior condiviso batte il core in qualita' assoluta dentro una classe
  stazionaria (pooling) ma e' accoppiato alla composizione del set — la
  protetta scambia qualita' media con invarianza. Numeri non confrontabili
  con neurobiologia.
- **Fase 15 (CL)**: `ReteEWC` e' un analogo concettuale in linguaggio di
  campo (forma chiusa + importanza `r^2` con decadimento), non l'EWC su
  gradienti di una rete torch — il confronto e' mele-con-mele sulla stessa
  Q, non una rivendicazione contro il CL neurale. Senza shift di classe
  tutti i metodi degradano poco (campi ~97% simili). EWC-lite ben tarato
  arriva a 0 distrutti (residuo ~0.008): la protetta vince sulla garanzia,
  non sul margine.
- **Fase 15 (plasticita')**: in regime facile il gate non filtra mai
  (tasso 1.0 fino a 800) — nessuna saturazione osservata senza vincolo di
  capacita'; il costo dello zero-forgetting e' memoria lineare + copertura
  che crolla a memoria limitata (12.5% con cap 100 su 800).
- **Fase 15 (OOD)**: retrieval MSE fragile (accuracy 1.0→0.125 con rumore);
  cue composizionali ambigue al centro (margine 6e-4); il completamento non
  e' ragionamento. Bug trovato e corretto durante gli smoke: gli id del set
  B sovrascrivevano A senza offset (slot chiave su id) — ora offset
  espliciti + test di non-collisione implicito (estremi mix corretti).
- **Fase 16 (sonno)**: il gist alza q (+0.04 su mix, +0.01-0.02 col rumore)
  ma e' pooling sfocato, non ragionamento; non risolve l'exact-match.
  Garanzia verificata: slot + nucleo bit-identici dopo il sonno.
- **Fase 16 (eviction)**: cancellazione misurata, rimasti invarianti; la
  politica per-Q pota i deboli (Q +0.009), eta/uso lasciano Q piatta.
  Proxy d'uso `id%3` documentato come proxy, non misura reale.
- **Fase 16 (transfer)**: BWT/FWT medi ~0 per tutti — la media-Q non vede il
  forgetting che il max/degrado (§6.10) mostra; unico segnale strutturale
  resta protetta = 0 esatto vs condivise con distrutti > 0.
- **Fase 16 (retrieval)**: validazione fit->val raddoppia al rumore forte
  ma dimezza sul pulito; coarse-to-fine non aggiunge. Retrieval ancora
  fragile — negativo onesto.

---

## 8. Stabilità numerica

- Passo stabile: `dt = 0.4 * dx^2 / (4 * max(D0, Dx, Dy))`.
- Per `N=96, L=1.0`: `dt ≈ 2.17e-04`.
- Aumentare `N` riduce `dt` come `1/N^2`: usare `T` piccoli in fase di test.
- SDE con Eulero-Maruyama: il rumore scala come `sqrt(dt)` (non `dt`);
  OU colorata a varianza unitaria con variabile locale (sicura cambiando `N`);
  `Fy` proiettata su `>= 0` (come il 1D che usa `clip`).
- CI: `.github/workflows/ci.yml` esegue fast test + full test con coverage
  (`--cov-fail-under=80`).

---

## 9. Risoluzione problemi

| Sintomo | Causa probabile | Soluzione |
|---|---|---|
| `ModuleNotFoundError: src` | lancio dalla cartella sbagliata | eseguire dalla radice del progetto |
| Simulazione lenta | `N` alto / `T` lungo | provare `--N 64 --T 0.1` |
| `matplot` senza finestre | ambiente headless | normale: backend `Agg`, aprire i PNG in `output/` |
| Figure mancanti | `output/` non scrivibile | verificare permessi, la cartella viene creata in automatico |

---

## 10. Riferimenti

- `frammetoveloce.md` — formalizzazione matematica completa (modello, equazioni, discussione, conclusioni).
- `Frammento_del_veloce_IT.pdf` — preprint in italiano.
  DOI: [10.5281/zenodo.22812892](https://doi.org/10.5281/zenodo.22812892).
- Tracce mnestiche citate: `zenodo.org/records/14534720`, `zenodo.org/records/17069503`.

---

## 11. Licenza e riuso

Licenza MIT (vedi `LICENSE`). Uso didattico / scientifico: citare l'autore
del modello (dr. Bulla Francesco) in derivazioni e pubblicazioni.

- `CITATION.cff`: metadati di citazione (DOI preprint:
  [10.5281/zenodo.22812892](https://doi.org/10.5281/zenodo.22812892)).
- Tracce mnestiche citate: `zenodo.org/records/14534720`,
  `zenodo.org/records/17069503`.
