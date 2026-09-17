# Frammento del Veloce

![coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)
![tests](https://img.shields.io/badge/tests-40_passed-brightgreen)
![python](https://img.shields.io/badge/python-3.13-blue)

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
│   ├── __init__.py        # export unificati Param / Params
│   ├── __main__.py        # CLI: 2d | 1d | demo | sweep | ablazione | all
│   ├── frammento_2d.py    # modello 2D toroidale (codice principale)
│   ├── frammento_1d.py    # simulatore 1D di riferimento
│   ├── demo_figure.py     # genera le 7 figure del preprint
│   └── studi.py           # sweep parametri + ablazione (CSV, md, fig08)
├── tests/                 # 38 test (fast + slow con --run-slow)
├── output/                # PNG/CSV/md generati (creata al primo run)
├── run.py                 # avvio rapido: python run.py [all]
├── pyproject.toml         # marker slow + config coverage
├── requirements.txt
└── README.md
```

| Modulo | Contenuto |
|---|---|
| `frammento_2d.py` | `Param`, `griglia`, `laplaciano`, `essenza`, `input_field` (Lissajous), `simula` (Eulero-Maruyama), `lyapunov`, `derivata_numerica`, `esperimento_diffusione`, `invariante_nv`, `turing_gy` (Gierer-Meinhardt), `metriche`, `fedelta`, `lfp_sintetico`, `riepilogo` |
| `frammento_1d.py` | `Params`, `Domain` (periodico / Neumann), `G0` / `GX` / `GY`, `History`, `FrammentoDelVeloce` (`project_novelty`, `project_mass`, `fidelity_test`, `report`), `plot` |
| `demo_figure.py` | `fig_tre_livelli`, `fig_evoluzione`, `fig_diagnostica`, `fig_metriche`, `fig_turing`, `fig_invariante`, `fig_lfp` |
| `studi.py` | `valuta`, `tempo_recupero` (twin experiment), `config_sweep`, `config_ablazione`, `main_sweep`, `main_ablazione`, `fig_ablazione` |

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

### 2.1 Equazioni 2D (modello principale)

```text
dF0/dt = D0 lap(F0)                                    (R_g0 = 0)
dFx/dt = Dx lap(Fx) + alpha (Fin - Fx) + kx (F0 - Fx)
dFy/dt = Dy lap(Fy) + beta Fy (1 - Fy/K) G + gamma G xi - ky Fy
```

- `G` = gate di compatibilita': `clip(Fx / (1.2 max(Fx)), 0, 1) * mask(g0)`.
- `mask(g0)` = supporto strutturale dell'essenza (`essenza > 1e-3`).
- `xi` = rumore bianco oppure colorato (Ornstein-Uhlenbeck).
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

### 2.4 Invarianza in forma normalizzata

La forma `n = v` e' un **principio qualitativo di bilanciamento** (n e' una
massa, v una velocita'/diffusivita': hanno dimensioni diverse). Nelle
simulazioni si usa la forma normalizzata nella scala fissata da g0:

```text
v_tilde = v / D_g0,    n = lambda_g * v_tilde
```

- `n` = massa totale del Frammento (normalizzata a 1).
- `v` = coefficiente di diffusione efficace misurato da `<r^2> = 4 D t`.
- `v_tilde` = velocita' adimensionale; `lambda_g` = costante di scala di g0.

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
- **Tempo di recupero**: twin experiment — gemello imperturbato vs gemello
  perturbato (bump gaussiano) in rilassamento deterministico; primo istante
  in cui l'energia della perturbazione scende sotto il 5% (`stato_iniziale`
  di `simula`).
- **LFP sintetico**: theta 6 Hz + gamma 45 Hz con ampiezza gamma modulata
  dalla novita' (analogia computazionale, vedi §7).

---

## 3. Requisiti

- Python 3.10+ (testato su 3.13)
- `numpy`, `matplotlib` (`scipy` elencata per estensioni future)

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
# veloci di default (36 test, <2 s; gli slow vengono skippati)
python -m pytest tests/ -q

# tutti, inclusi slow: demo completa + sweep/ablazione mini (~11 s)
python -m pytest tests/ -q --run-slow

# solo gli slow
python -m pytest tests/ -q --run-slow -m slow

# con coverage (XML in output/coverage.xml)
python -m pytest tests/ -q --run-slow --cov=src --cov-report=term-missing

# dentro Docker
docker run --rm --entrypoint python frammento-del-veloce:latest -m pytest tests/ -q --run-slow
```

Copertura: conservazione massa g0, decrescita di Lyapunov, stima di `D`
entro un fattore 2 (forma normalizzata `v_tilde`), range di qualita'/
continuita', fedelta' su gy, twin di recupero, LFP, Turing vincolato a `g0`,
conservazione massa 1D, `fidelity_test`, CLI 2d/1d, sweep/ablazione, import demo.

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
novita' gy (fine)  : 0.011568
gate gx medio      : 0.0935
Lyapunov V(0)      : 4.157018 | V(fine): 0.401803
frazione dV/dt<=0  : 100.0 %
g0: n = 1.0000  v = 0.037711  v~ = 0.7542  D_stimato = 0.037602 (vero 0.050000)
gx: n = 1.0000  v = 0.008630  v~ = 0.1726  D_stimato = 0.008629 (vero 0.010000)
gy: n = 1.0000  v = 0.001781  v~ = 0.0356  D_stimato = 0.001781 (vero 0.002000)
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
| `fig06_invariante_nv.png` | `<r^2> = 4Dt` + barre `v/D_g0`, `D_stim/D` |
| `fig07_lfp.png` | traccia LFP sintetica + spettro theta/gamma |
| `fig08_ablazione.png` | barre qualita'/continuita'/fedelta'/adattamento per baseline (da `ablazione`) |

La demo usa una configurazione dimostrativa (`N=80`, `T=0.22`) definita in
`src/demo_figure.py:main()` per tempi di calcolo contenuti.

### 5.4 Studi di robustezza (sweep + ablazione)

```bash
python -m src sweep --N 48 --T 0.15        # 7 configurazioni -> studio_parametri.csv/.md
python -m src ablazione --N 48 --T 0.15    # 6 baseline -> ablazione.csv/.md + fig08
```

Vedi §6 per le tabelle dei risultati.

---

## 6. Studi di robustezza

Condizioni: `N=48`, `T=0.15`, `seed=7`. `t_rec` con `+` = non recuperato
entro la finestra (`>0.30`): e' un limite inferiore onesto, non un recupero
istantaneo. CSV completi in `output/studio_parametri.csv` e `output/ablazione.csv`.

### 6.1 Sweep dei parametri

| config | massa g0 | qualita' | continuita' | fedelta' (gy) | max dV/dt | dV/dt<=0 | novita' | t_rec |
|---|---|---|---|---|---|---|---|---|
| A base (stimolo, bianco) | 1.0000 | 0.4877 | 0.9995 | 1.0000 | -6.83 | 100% | 0.0121 | >0.30 |
| B rumore nullo | 1.0000 | 0.4877 | 0.9995 | 1.0000 | -6.83 | 100% | 0.0121 | >0.30 |
| C rumore alto (γ=0.15) | 1.0000 | 0.4877 | 0.9995 | 0.9999 | -6.83 | 100% | 0.0121 | >0.30 |
| D rumore OU colorato | 1.0000 | 0.4877 | 0.9995 | 0.9994 | -6.83 | 100% | 0.0121 | >0.30 |
| E creativita' alta (β=2.0) | 1.0000 | 0.4877 | 0.9995 | 1.0000 | -6.83 | 100% | 0.0123 | >0.30 |
| F rilassamento | 1.0000 | 0.4872 | 0.9995 | 1.0000 | -6.80 | 100% | 0.0121 | >0.30 |
| G adattamento forte (α=8.0) | 1.0000 | **0.2554** | 0.9994 | 1.0000 | -5.45 | 100% | 0.0121 | 0.243 |

Lettura: il sistema e' stabile in tutte le configurazioni (`dV/dt<=0` al
100%, massa g0 esatta); la qualita' crolla solo con adattamento eccessivo
(G: l'input trascina gx lontano dall'essenza); il rumore OU impatta la
fedelta' di gy piu' del bianco a pari intensita'; G recupera piu' in fretta
(accoppiamento forte = richiamo piu' rapido).

### 6.2 Ablazione dei livelli

| config | massa g0 | qualita' | novita' | fedelta' | dV/dt<=0 | t_rec | adattamento |
|---|---|---|---|---|---|---|---|
| 1 solo diffusione | 1.0000 | 0.7088 | 0.0119 | 1.0000 | 100% | >0.30 | -0.04 |
| 2 g0+gx (senza gy) | 1.0000 | 0.4877 | 0.0119 | 1.0000 | 100% | >0.30 | -0.04 |
| 3 completo | 1.0000 | 0.4877 | 0.0121 | 1.0000 | 100% | >0.30 | -0.04 |
| 4 senza vincolo (ky=0, K=5) | 1.0000 | 0.4877 | **0.0125** | 1.0000 | 100% | >0.30 | -0.04 |
| 5 senza rumore | 1.0000 | 0.4877 | 0.0121 | 1.0000 | 100% | >0.30 | -0.04 |
| 6 rumore eccessivo (γ=0.25) | 1.0000 | 0.4877 | 0.0121 | 0.9998 | 100% | >0.30 | -0.04 |

Lettura: g0 conserva la struttura da solo (qualita' massima senza gx);
gx introduce l'adattamento al prezzo della somiglianza all'essenza;
gy aggiunge novita' (+2% vs config 2); togliere il vincolo fa crescere la
novita' (+3% vs completo: il vincolo trattiene davvero); il rumore
moderato non degrada nulla, quello eccessivo intacca la fedelta' di gy.

## 7. Limiti e natura del modello

- **Teorico-computazionale**, non validato su dati biologici (nessun
  EEG/fMRI/comportamentale reale).
- **Sensibile ai parametri**: oltre una soglia di adattamento (α≈8) la
  qualita' si dimezza; i vincoli di gy sono necessari (config 4).
- **Spazio cognitivo astratto**: nessuna interpretazione neuroanatomica
  diretta di g0/gx/gy per ora.
- **Rumore additivo nel tasso**: il termine `gamma*G*xi` entra nella
  reazione senza scaling `sqrt(dt)`, quindi il suo effetto per-step e'
  O(dt)-piccolo; la fedelta' di gy resta >0.999 in tutte le config.
- **Recupero lento**: una perturbazione localizzata non rientra al 5% entro
  `t=0.30` in quasi tutte le config (scala di rilassamento ~1/kappa_x).
- **Orizzonte breve**: sweep/ablazione a `T=0.15`; comportamenti su tempi
  lunghi e pattern di Turing completi restano da esplorare.

---

## 8. Stabilità numerica

- Passo stabile: `dt = 0.4 * dx^2 / (4 * max(D0, Dx, Dy))`.
- Per `N=96, L=1.0`: `dt ≈ 2.17e-04`.
- Aumentare `N` riduce `dt` come `1/N^2`: usare `T` piccoli in fase di test.
- Il rumore colorato usa una variabile OU locale (sicura anche cambiando `N` tra run diversi).

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
- Tracce mnestiche citate: `zenodo.org/records/14534720`, `zenodo.org/records/17069503`.

---

## 11. Licenza e riuso

Uso didattico / scientifico. Citare l'autore del modello (dr. Bulla Francesco)
in derivazioni e pubblicazioni.
