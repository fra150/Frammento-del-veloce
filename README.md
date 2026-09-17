# Frammento del Veloce

![coverage](https://img.shields.io/badge/coverage-94%25-brightgreen)
![tests](https://img.shields.io/badge/tests-47_passed-brightgreen)
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
├── tests/                 # 47 test (43 fast + 4 slow con --run-slow)
├── output/                # PNG/CSV/md generati (creata al primo run)
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
# veloci di default (43 test, ~4 s; gli slow vengono skippati)
python -m pytest tests/ -q

# tutti, inclusi slow: demo completa + sweep/ablazione mini (~24 s)
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

## 7. Limiti e natura del modello

- **Teorico-computazionale**, non validato su dati biologici (nessun
  EEG/fMRI/comportamentale reale).
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
