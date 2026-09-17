# Frammento del Veloce

![coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)
![tests](https://img.shields.io/badge/tests-32_passed-brightgreen)
![python](https://img.shields.io/badge/python-3.13-blue)

Libreria di simulazione numerica per la diffusione dinamica della memoria.

Modello a tre livelli geometrici:

- **g0** — essenza perfetta / invariante (`R = 0`, solo diffusione conservativa)
- **gx** — operativita' vincolata agli input (accoppiamento con `x(t)`)
- **gy** — novita' controllata (reazione non lineare + rumore, vincolata da `g0` e `gx`)

Riferimento concettuale: reazione-diffusione con misura invariante, funzione di
Lyapunov e invarianza `n = v` (numero di elementi = velocita' di diffusione,
a meno della costante di scala fissata dal livello `g0`).

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
│   ├── __main__.py        # CLI unificata: python -m src <2d|1d|demo|all>
│   ├── frammento_2d.py    # modello 2D toroidale (codice principale)
│   ├── frammento_1d.py    # simulatore 1D di riferimento
│   └── demo_figure.py     # genera le 7 figure del preprint
├── output/                # PNG generati (creata al primo run)
├── run.py                 # avvio rapido: python run.py [all]
├── requirements.txt
└── README.md
```

| Modulo | Contenuto |
|---|---|
| `frammento_2d.py` | `Param`, `griglia`, `laplaciano`, `essenza`, `input_field` (Lissajous), `simula` (Eulero-Maruyama), `lyapunov`, `derivata_numerica`, `esperimento_diffusione`, `invariante_nv`, `turing_gy` (Gierer-Meinhardt), `metriche`, `fedelta`, `lfp_sintetico`, `riepilogo` |
| `frammento_1d.py` | `Params`, `Domain` (periodico / Neumann), `G0` / `GX` / `GY`, `History`, `FrammentoDelVeloce` (`project_novelty`, `project_mass`, `fidelity_test`, `report`), `plot` |
| `demo_figure.py` | `fig_tre_livelli`, `fig_evoluzione`, `fig_diagnostica`, `fig_metriche`, `fig_turing`, `fig_invariante`, `fig_lfp` |

---

## 2. Modello matematico (sintesi)

### 2.1 Equazioni 2D (modello principale)

```text
dF0/dt = D0 lap(F0)                                    (R_g0 = 0)
dFx/dt = Dx lap(Fx) + alpha (Fin - Fx) + kx (F0 - Fx)
dFy/dt = Dy lap(Fy) + beta Fy (1 - Fy/K) G + gamma G xi - ky Fy
```

- `G` = gate di compatibilita': `clip(Fx / (1.2 max(Fx)), 0, 1) * mask(g0)`.
- `mask(g0)` = supporto strutturale dell'essenza (`essenza > 1e-3`).
- `xi` = rumore bianco oppure colorato (Ornstein-Uhlenbeck).
- Dominio: toro `[0, L)^2`, laplaciano periodico con `np.roll`.

### 2.2 Funzione di Lyapunov

```text
V = w0 ||F0 - F0*||^2 + wx ||Fx - F0||^2 + wy ||Fy||^2
```

con `F0*` = media spaziale di `F0` (equilibrio omogeneo).
Stabilita' verificata come frazione di `dV/dt <= 0` lungo la traiettoria.

### 2.3 Invariante n = v

- `n` = massa totale del Frammento (normalizzata a 1).
- `v` = coefficiente di diffusione efficace misurato da `<r^2> = 4 D t`.
- `v_tilde = v / D_g0` = velocita' adimensionale nella scala fissata da `g0`.

### 2.4 Novita' controllata (Turing in gy)

Attivatore-inibitore di Gierer-Meinhardt confinato nel supporto di `g0`:

```text
ra = rho a^2 / (h + eps) - mu_a a
rh = rho a^2 - mu_h h
```

### 2.5 Metriche

- **Qualita'**: `1 - ||Fx - essenza|| / ||essenza||`.
- **Continuita'**: coseno tra frame consecutivi di `gx`.
- **Fedelta'**: correlazione tra ricordo con / senza rumore.
- **LFP sintetico**: theta 6 Hz + gamma 45 Hz con ampiezza gamma modulata dalla novita'.

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
# veloci di default (29 test, <2 s; gli slow vengono skippati)
python -m pytest tests/ -q

# tutti, inclusa la demo completa con 7 figure (~10 s)
python -m pytest tests/ -q --run-slow

# solo gli slow
python -m pytest tests/ -q --run-slow -m slow

# con coverage (XML in output/coverage.xml)
python -m pytest tests/ -q --run-slow --cov=src --cov-report=term-missing

# dentro Docker
docker run --rm --entrypoint python frammento-del-veloce:latest -m pytest tests/ -q --run-slow
```

Copertura: conservazione massa g0, decrescita di Lyapunov, stima di `D`
entro un fattore 2 (`n = v`), range di qualita'/continuita', fedelta',
LFP, Turing vincolato a `g0`, conservazione massa 1D, `fidelity_test`,
parsing CLI e import demo.

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

La demo usa una configurazione dimostrativa (`N=80`, `T=0.22`) definita in
`src/demo_figure.py:main()` per tempi di calcolo contenuti.

---

## 6. Stabilità numerica

- Passo stabile: `dt = 0.4 * dx^2 / (4 * max(D0, Dx, Dy))`.
- Per `N=96, L=1.0`: `dt ≈ 2.17e-04`.
- Aumentare `N` riduce `dt` come `1/N^2`: usare `T` piccoli in fase di test.
- Il rumore colorato usa una variabile OU locale (sicura anche cambiando `N` tra run diversi).

---

## 7. Risoluzione problemi

| Sintomo | Causa probabile | Soluzione |
|---|---|---|
| `ModuleNotFoundError: src` | lancio dalla cartella sbagliata | eseguire dalla radice del progetto |
| Simulazione lenta | `N` alto / `T` lungo | provare `--N 64 --T 0.1` |
| `matplot` senza finestre | ambiente headless | normale: backend `Agg`, aprire i PNG in `output/` |
| Figure mancanti | `output/` non scrivibile | verificare permessi, la cartella viene creata in automatico |

---

## 8. Riferimenti

- `frammetoveloce.md` — formalizzazione matematica completa (modello, equazioni, discussione, conclusioni).
- `Frammento_del_veloce_IT.pdf` — preprint in italiano.
- Tracce mnestiche citate: `zenodo.org/records/14534720`, `zenodo.org/records/17069503`.

---

## 9. Licenza e riuso

Uso didattico / scientifico. Citare l'autore del modello (dr. Bulla Francesco)
in derivazioni e pubblicazioni.
