# Atención heterogénea en Vision Transformers pequeños

[English](README.md)

Código, configuraciones y registros de los experimentos del artículo *Inductive Biases, Not Similarity Functions: A Controlled Evaluation of Performance, Scale and Calibration of Heterogeneous Attention in Small Vision Transformers* (Martínez-Marín, Hernández-Nolasco y Zacarías-Morales, enviado).

**En una frase:** en un ViT de 119 k parámetros entrenado desde cero, combinar dos funciones de similitud dentro de una cabeza (producto punto + atención aditiva) no supera a un control de igual capacidad. En cambio, añadir un sesgo inductivo que la atención no tiene, una convolución depthwise 5×5 sobre los valores de cada cabeza, aporta unos 5 puntos en CIFAR-10. La heterogeneidad útil está en los sesgos inductivos, no en las funciones de similitud.

## Contenido

| Ruta | Qué es |
|---|---|
| `src/ham/` | Modelo (`models/`: ViT, ramas de atención, fusiones, términos de salida, ResNet), entrenamiento (`train.py`), evaluación (`evaluate.py`), análisis posterior (`analyze.py`) y visualización |
| `configs/` | Un YAML por variante (`<conjunto>_<variante>.yaml`), que hereda de `_<conjunto>.yaml` y de `base.yaml` |
| `scripts/run_paper.sh` | Todas las fases de experimentos del artículo (`fase1` … `fase18`) |
| `scripts/ieee_tables.py`, `scripts/tables_en.py`, `scripts/paper_figures.py` | Tablas y figuras del artículo, generadas a partir de los registros |
| `scripts/costos.py` → `results/costos.json` | Costos independientes del hardware (MACs, FLOPs, memoria) y latencia en CPU |
| `records/runs.tar.gz` | Registros de los 830 runs usados en el artículo (ver abajo) |
| `paper/ieee_en/` | Fuentes LaTeX del artículo (*preprint*, formato IEEE) |
| `tests/` | Pruebas (`pytest`) |

## Instalación

Requiere [uv](https://docs.astral.sh/uv/). `uv.lock` fija las versiones exactas usadas: Python 3.13 y PyTorch 2.14.1 con CUDA 13.0.

```bash
uv sync
uv run pytest
```

## Reproducir las tablas sin entrenar

Cada run guardó lo siguiente:
- su configuración resuelta;
- el commit del código y el entorno;
- las métricas por época;
- la evaluación en test;
- el análisis posterior.

`records/runs.tar.gz` contiene esos archivos (`config.resolved.yaml`, `provenance.json`, `metrics.json`, `test.json`, `analysis.json`) de los 830 runs, con la estructura de directorios que esperan los scripts. No incluye los pesos (`best.pt`), las predicciones ni los índices del split; el split se regenera a partir de `data.split_seed`.

```bash
tar xzf records/runs.tar.gz                     # crea runs/, runs_seeds/, runs_corta/, runs_noaug/
uv run python scripts/ieee_tables.py            # tablas en español (paso intermedio) -> paper/ieee/tablas/
uv run python scripts/tables_en.py              # tablas en inglés -> paper/ieee_en/tablas/
git diff --stat paper/ieee_en/tablas            # debe salir vacío: idénticas a las del artículo
```

Las figuras que sólo necesitan los registros (`curvas.pdf`, `semillas.pdf`, `kernel.pdf`, `mecanismo.pdf`) se regeneran con:

```bash
uv run python scripts/paper_figures.py --lang en --out paper/ieee_en --sin-mapas
```

Los mapas de atención (`mapas.png`) y su tabla de similitud (`paper/tablas/mapas.tex`, que se incluye tal cual) necesitan los pesos.

## Entrenar

Los datos se buscan en `data/` (`data.root`):
- CIFAR-10, CIFAR-100 y SVHN se descargan solos con torchvision.
- PlantVillage (`data/plantvillage/{train,val}/<clase>/`) y Food-101 (`data/food-101/{images,meta}/`) se descargan aparte.

```bash
uv run python -m ham.train --config configs/cifar10_dot_local_k5.yaml --seed 0
scripts/run_seeds.sh "0 1 2 3 4" configs/cifar10_dot.yaml configs/cifar10_dot_local_k5.yaml
scripts/run_paper.sh fase12
uv run python -m ham.analyze --run runs/cifar10/cifar10_dot_local_k5/seed0
```

Cualquier campo se sobrescribe desde la línea de comandos (`train.epochs=1`); las claves desconocidas producen error.

**Protocolo.**
- División 45k/5k/10k en CIFAR-10.
- Se conserva el modelo con la mejor exactitud de validación, y el test se evalúa una sola vez.
- El entrenamiento es determinista: con la misma semilla, código y entorno, las métricas se reproducen bit a bit. Por eso las semillas se parean entre variantes.
- El artículo usó una NVIDIA RTX 5080. En otra GPU u otra versión de CUDA puede haber diferencias pequeñas.

La tabla de variantes principales está en [README.md](README.md#main-variants).

## Compilar el artículo

```bash
cd paper/ieee_en && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Pesos, cita y licencia

- **Pesos:** los de todos los runs (≈ 1.1 GB) se publicarán como *release* o en Zenodo. <!-- PENDIENTE: enlace -->
- **Cita:** ver `CITATION.cff`. Archivado en Zenodo: [doi:10.5281/zenodo.24452704](https://doi.org/10.5281/zenodo.24452704).
- **Licencias:** el código es MIT (`LICENSE`); los registros y las fuentes del artículo, CC BY 4.0 (`LICENSE-records`).

Este repositorio se exporta del repositorio de trabajo del autor; no incluye las notas internas de trabajo.
