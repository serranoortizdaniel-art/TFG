# Simulador de ataques y defensas en Aprendizaje Federado

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![Flower](https://img.shields.io/badge/Flower-1.31-3E8E7E)
![Tests](https://img.shields.io/badge/tests-39%20passed-success)
![Config](https://img.shields.io/badge/config-YAML%20declarativo-8A2BE2)

Entorno de simulación de **Aprendizaje Federado horizontal con clientes maliciosos**, construido sobre Flower y PyTorch. Permite definir experimentos completos en YAML (dataset, particionado, ataque, defensa, semilla), ejecutarlos de forma reproducible en una sola máquina y obtener automáticamente métricas por ronda, normas de las actualizaciones de cada cliente, matrices de confusión, figuras y tablas comparativas.


## Características

- **Simulación federada completa** cliente-servidor con participación total por ronda, clientes virtuales sobre Ray y evaluación centralizada del modelo global tras cada agregación.
- **Ataques de envenenamiento** enchufables, con dos puntos de intervención: sobre las etiquetas locales antes de entrenar (*data poisoning*) o sobre la actualización antes de enviarla (*model poisoning*).
- **Cuatro agregadores** tras una interfaz común: FedAvg, mediana coordinada, Trimmed Mean y Krum, con política de parámetros fija u *oracle*.
- **Instrumentación por cliente y ronda**: norma L2 de cada actualización con marca de malicioso (solo en el registro: la defensa trabaja a ciegas), tiempos de ronda y de agregación.
- **Particionado IID y non-IID** (Dirichlet con α configurable y tamaño mínimo por cliente), idéntico entre ejecuciones con la misma semilla.
- **Configuración declarativa**: cada experimento es un YAML pequeño sobre valores por defecto validados; cualquier campo se puede sobreescribir desde la CLI sin editar ficheros.
- **Ejecución por lotes reanudable**: un lanzador genera la malla completa de escenarios, se puede interrumpir en cualquier momento y no repite ejecuciones completadas.
- **Análisis automatizado**: consolidación con media ± σ y degradaciones frente a líneas base, curvas, matrices de confusión, normas y distribuciones de clases; tablas en LaTeX y Markdown.
- **Reproducibilidad estricta**: una semilla controla todas las fuentes de aleatoriedad; la configuración efectiva se guarda junto a cada resultado.
- **39 tests unitarios** (agregadores con vectores sintéticos de resultado conocido, ataques, particionado, semillas).

## Cómo funciona

Cada ejecución sigue el mismo flujo:

```
YAML + defaults ─► semilla global ─► dataset en memoria ─► particionado IID/Dirichlet
        │
        ▼
Flower (Ray) crea los clientes virtuales
        │  cliente benigno:  entrena y envía Δ
        │  cliente malicioso: ataque sobre etiquetas (antes) o sobre Δ (después)
        ▼
Strategy propia: registra la norma L2 de cada Δ ─► delega en el agregador configurado
        ▼
modelo global ─► evaluación centralizada por ronda ─► CSV + JSON estructurados
```

Decisiones de diseño relevantes:

- **Flower no concentra la lógica**: orquesta rondas y recursos, pero ataques, agregadores y métricas viven en módulos propios, testeables sin levantar ninguna simulación.
- **El agregador nunca sabe qué clientes son maliciosos**; esa marca existe solo en la instrumentación, como ocurriría en un despliegue real.
- **Los clientes maliciosos se seleccionan de forma determinista** a partir de la semilla y la fracción configurada: dos escenarios comparados con la misma semilla difieren solo en lo que se quiere comparar.
- Las actualizaciones se ordenan por id de cliente antes de agregar, de modo que agregadores sensibles a empates (Krum) son estables frente al orden no determinista de Ray.
- Los datasets se normalizan una sola vez y residen en memoria como tensores, lo que elimina la preparación de lotes como cuello de botella.

## Instalación

La simulación se ejecuta en **WSL/Ubuntu** (o Linux nativo): `flwr[simulation]` depende de Ray, sin wheel para Python 3.13 en Windows nativo; el entorno validado es WSL2 + Python 3.12.

```bash
python3 -m venv ~/.venvs/pfg-flower
~/.venvs/pfg-flower/bin/python3 -m pip install -r requirements-flower-wsl.txt
```

`requirements.txt` fija versiones exactas de todas las dependencias; `requirements-flower-wsl.txt` añade Flower + Ray. En una máquina **sin GPU NVIDIA**, sustituir `torch==2.12.0+cu126` y `torchvision==0.27.0+cu126` por `torch==2.12.0` y `torchvision==0.27.0`. Los datasets (MNIST, Fashion-MNIST, CIFAR-10) se descargan solos en `data/` la primera vez.

## Inicio rápido

```bash
# 1. Validación del entorno (problema sintético, tarda segundos)
~/.venvs/pfg-flower/bin/python3 scripts/validate_flower.py

# 2. Un experimento: label flipping + Krum en non-IID
~/.venvs/pfg-flower/bin/python3 run_experiment.py --config configs/fashion_mnist_label_flipping_krum_non_iid.yaml

# 3. Malla completa de escenarios (reanudable)
~/.venvs/pfg-flower/bin/python3 run_all.py

# 4. Consolidados, figuras y tablas a partir de lo que haya en results/raw
~/.venvs/pfg-flower/bin/python3 run_analysis.py
```

## Configuración

Un experimento se declara con los campos que difieren de los valores por defecto (`configs/example_completo_comentado.yaml` documenta todos):

```yaml
dataset: fashion_mnist        # mnist | fashion_mnist | cifar10
num_clients: 10
num_rounds: 30
partition_type: dirichlet     # iid | dirichlet
dirichlet_alpha: 0.5
attack_type: label_flipping   # none | label_flipping | sign_flipping | scaling
malicious_fraction: 0.3
aggregator: krum              # fedavg | median | trimmed_mean | krum
aggregator_params:
  policy: fixed               # fixed | oracle
seed: 42
```

Cualquier parámetro, incluidos los anidados, se sobreescribe desde la CLI con notación de puntos:

```bash
~/.venvs/pfg-flower/bin/python3 run_experiment.py --config configs/base_fedavg_iid_fmnist.yaml \
    --seed 123 --override attack_type=sign_flipping malicious_fraction=0.2 aggregator=median
```

El nombre canónico de cada ejecución se construye con sus factores (`fashion_mnist_dir0.5_sign_flipping_m20_krum`) y cada semilla vive en su subcarpeta, así que agregar resultados entre semillas es directo.

## Ataques y agregadores disponibles

| Ataque | Tipo | Parámetros principales | En la malla por defecto |
|---|---|---|:---:|
| `label_flipping` (`flip_mode=all_to_next`) | datos | flip c→(c+1) mod C al 100 % | ✔ |
| `label_flipping` (`flip_mode=targeted`) | datos | `source_class`, `target_class`, `flip_fraction` | ✘ |
| `sign_flipping` | modelo | `gamma` (Δ → −γΔ) | ✔ (γ=1) |
| `scaling` | modelo | `scale_factor` (Δ → kΔ) | ✔ (k=10; 20 % y 30 %) |

| Agregador | Regla | Parámetros |
|---|---|---|
| `fedavg` | media ponderada por nº de muestras | — |
| `median` | mediana por coordenada | — |
| `trimmed_mean` | media recortando los β·K valores extremos por coordenada | `trim_beta` |
| `krum` | selecciona la actualización con menor suma de distancias a sus K−f−2 vecinos | `krum_f` |

Con `policy: fixed` los parámetros de defensa usan una cota conservadora constante (`krum_f=3`, `trim_beta=0.3` para K=10); con `policy: oracle` se ajustan a la fracción real de maliciosos de cada escenario. La malla principal ejecuta la política fija con los cuatro agregadores y añade la política *oracle* para Krum y Trimmed Mean, las dos defensas que dependen de estos parámetros.

La variante marcada con ✘ (*flip* dirigido), el soporte de CIFAR-10 y escenarios alternativos como `dirichlet_alpha=0.1` o `num_clients=20` están implementados y listos para usar vía configuración, aunque no forman parte de la malla que lanza `run_all.py`. La malla sí incluye *scaling*, la política `oracle` y fracciones de maliciosos de hasta el 40 %:

```bash
# Ejemplo manual: scaling x10 al 30 % contra Trimmed Mean oracle en non-IID
~/.venvs/pfg-flower/bin/python3 run_experiment.py --config configs/base_fedavg_iid_fmnist.yaml \
    --override attack_type=scaling attack_params.scale_factor=10.0 malicious_fraction=0.3 \
    partition_type=dirichlet aggregator=trimmed_mean aggregator_params.policy=oracle
```

## Ejecución por lotes

`run_all.py` genera y ejecuta la matriz completa sobre Fashion-MNIST. Para cada par (partición, agregador) evalúa once escenarios: uno sin ataque, *label flipping* y *sign flipping* con 10 %, 20 %, 30 % y 40 % de clientes maliciosos, y *scaling* con 20 % y 30 %. Esto produce:

- **88 configuraciones con política fija**: 2 particiones × 4 agregadores × 11 escenarios.
- **44 configuraciones adicionales con política oracle**: 2 particiones × 2 agregadores parametrizados × 11 escenarios.
- **132 configuraciones y 396 ejecuciones** al repetir cada una con las semillas 42, 123 y 2024.

```bash
~/.venvs/pfg-flower/bin/python3 run_all.py --dry-run        # imprime el plan sin ejecutar
~/.venvs/pfg-flower/bin/python3 run_all.py --filter krum --limit 5   # subconjuntos
~/.venvs/pfg-flower/bin/python3 run_all.py --device cuda --client-device cuda --client-cpus 4 --client-gpus 1 --ray-cpus 4
```

Es **interrumpible y reanudable** (Ctrl+C cuando haga falta): las ejecuciones cuyo `metadata.json` está en estado `completed` no se repiten al relanzar. El progreso, con duración y accuracy de cada ejecución y una ETA, queda en `results/run_all_log.csv`.

Para WSL hay un lanzador en segundo plano que evita lanzar dos mallas a la vez:

```bash
bash scripts/ejecutar_matriz_experimental.sh gpu    # o cpu
PFG_FLOWER_VENV=/otra/ruta/venv bash scripts/ejecutar_matriz_experimental.sh gpu
```

### GPU (perfil para 4 GB de VRAM)

Por defecto los clientes entrenan en CPU, la opción más estable para tiradas largas. En una GPU pequeña conviene limitar Ray a un único actor con GPU:

```bash
~/.venvs/pfg-flower/bin/python3 run_experiment.py --config <config>.yaml \
    --override device=cuda flower.client_device=cuda flower.num_cpus=4.0 flower.num_gpus=1.0 flower.ray_init_args.num_cpus=4
```

### Referencia centralizada

Entrena la misma arquitectura sobre el dataset completo, sin federar, con las mismas métricas; útil como cota superior al interpretar cualquier escenario federado:

```bash
~/.venvs/pfg-flower/bin/python3 train_centralized.py --dataset fashion_mnist --epochs 10
```

## Salidas

Cada ejecución crea `results/raw/<experimento>/seed_<n>/`:

| Fichero | Contenido |
|---|---|
| `config.yaml` | configuración efectiva completa (reproducción exacta) |
| `metrics.csv` | por ronda: accuracy, loss, macro-F1, accuracy por clase, tiempos de ronda y de agregación |
| `update_norms.csv` | norma L2 de la actualización de cada cliente y ronda, con marca de malicioso |
| `class_distribution.csv` | muestras por clase y cliente de la partición real |
| `confusion_matrix.csv` | matriz de confusión del modelo final |
| `metadata.json` | estado, duración, hardware, clientes maliciosos, parámetros efectivos de la defensa |

`run_analysis.py` lee todo lo disponible en `results/raw/` y produce:

- `results/processed/`: métricas de todas las rondas en formato largo y resumen final con media ± σ por configuración y política, además de la degradación frente a dos referencias (FedAvg limpio y el mismo agregador limpio con la misma política).
- `plots/`: curvas de accuracy y loss para la política fija (media con banda de desviación entre semillas y línea base limpia), matrices de confusión promediadas, normas benignos/maliciosos por ronda y distribución de clases por cliente.
- `results/processed/tables/`: tablas comparativas en LaTeX y Markdown, separadas entre política fija y *oracle*.

Los módulos de `src/analysis/` también funcionan por separado (`python -m src.analysis.plot_accuracy`, etc.). Los CSV se vuelcan a disco al final de cada ronda: una ejecución interrumpida conserva todas las rondas completadas.

## Tests

```bash
~/.venvs/pfg-flower/bin/python3 -m pytest tests/
```

39 tests cubren los agregadores con vectores sintéticos de salida conocida (mediana exacta ante extremos, recorte de Trimmed Mean, Krum nunca selecciona un outlier evidente, equivalencia de todos con la media ante actualizaciones idénticas), los ataques (no mutan el dataset original, reproducibles por semilla), el particionado (todas las muestras exactamente una vez, tamaño mínimo, mayor heterogeneidad de Dirichlet) y el control de semillas y configuración.

## Reproducibilidad

Una única semilla por configuración controla la inicialización del modelo, el particionado, la selección de clientes maliciosos, el barajado de lotes y cualquier muestreo de los ataques. Las sub-semillas por componente se derivan con SHA-256 (`src/utils/seed.py`), de modo que cada fuente de aleatoriedad es independiente y estable entre plataformas; cuDNN se configura en modo determinista (coste en velocidad asumido).

## Rendimiento y hardware de referencia

Validado en Windows 11 + WSL2 Ubuntu 24.04, Python 3.12.3, Flower 1.31.0, Ray 2.55.1, PyTorch 2.12.0+cu126, NVIDIA RTX 3050 Ti Laptop (4 GB).

- Matriz completa (396 ejecuciones × 30 rondas, perfil GPU conservador): **~9.7 h** de pared, ~88 s por ejecución, ~2.8 s por ronda.
- Coste medio de la agregación: 26 ms (FedAvg), 32 ms (Trimmed Mean), 62 ms (mediana), 97 ms (Krum).
- Validaciones Flower sobre MNIST (4 clientes, 2 rondas): ~45 s en CPU.

## Estructura del proyecto

```
.
├── configs/                  # configuraciones YAML de experimentos
├── src/
│   ├── datasets/             # carga en memoria y particionado IID/Dirichlet
│   ├── models/               # CNN (28x28 y CIFAR) y bucle de entrenamiento
│   ├── federated/            # cliente/Strategy Flower, simulación y parámetros
│   ├── attacks/              # none, label_flipping, sign_flipping, scaling
│   ├── aggregators/          # fedavg, median, trimmed_mean, krum
│   ├── logging/              # CSV de métricas/normas y metadatos por ejecución
│   ├── analysis/             # consolidación, figuras y tablas
│   └── utils/                # config, semillas, métricas, hardware
├── tests/                    # 39 tests unitarios (pytest)
├── scripts/
│   ├── validate_flower.py    # validación técnica mínima de Flower
│   └── ejecutar_matriz_experimental.sh  # lanzador WSL de la malla completa
├── run_experiment.py         # un experimento desde YAML
├── run_all.py                # malla completa reanudable
├── run_analysis.py           # todos los productos de análisis
├── train_centralized.py      # referencia centralizada
├── requirements.txt          # dependencias base (versiones exactas)
├── requirements-flower-wsl.txt  # + Flower/Ray para WSL/Linux
└── results/, plots/, data/, checkpoints/   (generados; fuera de git)
```

## Limitaciones

- Es una **simulación local**: clientes virtuales en una máquina, sin latencias de red, caídas ni heterogeneidad de dispositivos.
- **Participación completa** en todas las rondas (no hay muestreo parcial de clientes).
- Los ataques implementados son variantes **no adaptativas**: ninguno se optimiza conociendo la regla de agregación.
- Usa la **API programática de simulación de Flower 1.31**, hoy marcada como obsoleta en favor de las Flower Apps; las versiones están fijadas para que esto no afecte a la reproducibilidad.
- No incluye ataques de inferencia/privacidad ni defensas con estado entre rondas.

## Autoría

Desarrollado por **Daniel Serrano Ortiz**.
