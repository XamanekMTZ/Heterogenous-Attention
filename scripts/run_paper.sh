#!/usr/bin/env bash
# Corridas del paper, por fases. Reanudable: si se interrumpe, relanzar la misma fase.
#   scripts/run_paper.sh check            # entorno, GPU y datos
#   scripts/run_paper.sh bench            # 3 épocas por variante -> runs_bench/ (para estimar tiempos)
#   scripts/run_paper.sh fase1|fase2|fase3|fase4|fase4_food|fase5|fase6|fase7|fase8|fase8_semillas
#   scripts/run_paper.sh fase_salidas|fase_salidas_noaug   # (alias fase9) espejo de las fusiones de scores, independiente de fase8
#   scripts/run_paper.sh fase10|fase10_corta|fase10_semillas|fase10_pv   # enriquecimientos de la salida de las cabezas
#   scripts/run_paper.sh fase11           # controles de dot_local: kernel 1 y 5, ham_concat + local
#   scripts/run_paper.sh fase12           # dot_local_k5 con 16 semillas y en PlantVillage, dot_local_k7
#   scripts/run_paper.sh fase13           # ham_concat_local_k5: ¿la rama global suma sobre dot_local_k5?
#   scripts/run_paper.sh fase14|fase14_extra   # control de capacidad dot_wide_local_k5 y ham_concat_local_k5 a 16 semillas
#   scripts/run_paper.sh fase15|fase15b|fase15_pv   # otras variantes HAM con local 5x5 (INFORME_HAM.md, 10.15)
#   scripts/run_paper.sh fase16           # ham_heads_local_k5 y additive_local_k5 a 16 semillas (INFORME_HAM.md, 10.16)
#   scripts/run_paper.sh fase17|fase17_pv # la tríada contra su control de capacidad: semillas 16-47 y PlantVillage
#   scripts/run_paper.sh fase18           # revisión: mezcla (CutMix, aumento fuerte), baselines, CIFAR-100/SVHN, datos, escala
#   scripts/run_paper.sh fase18_mezcla|fase18_baselines|fase18_datos|fase18_fraccion|fase18_escala
#   scripts/run_paper.sh todo             # fases 1, 5, 2 y 3 en ese orden
#   scripts/run_paper.sh prelim           # preliminares: 20 épocas, 3 semillas -> runs_prelim/ + results/prelim/
#   scripts/run_paper.sh analizar [runs]            # ham.analyze en cada run terminado (sólo inferencia)
#   scripts/run_paper.sh reporte [runs] [destino]   # analizar + tablas, curvas, estadística y mapas de atención
# Variables: SEEDS (default "0 1 2 3 4"); MEJORES y MEJORES_PV (fase10_semillas y fase10_pv).
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2 3 4"}
C=configs

fase1() {  # tabla principal, CIFAR-10
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_{dot,additive,ham_concat,ham_weighted,ham_sum}.yaml
}
fase2() {  # ablaciones, CIFAR-10
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_ham_pairwise.yaml $C/cifar10_ham_weighted_learn.yaml
  for lam in 0.5 0.7; do
    scripts/run_seeds.sh "$SEEDS" $C/cifar10_ham_weighted.yaml -- \
      "model.attention.fusion_kwargs.lambda_=$lam" "name=cifar10_ham_weighted_l${lam/./}"
  done
}
fase3() {  # generalización: fusiones de salida (caben en GPUs modestas)
  scripts/run_seeds.sh "$SEEDS" $C/plantvillage_{dot,additive,ham_concat}.yaml
  scripts/run_seeds.sh "$SEEDS" $C/food101_{dot,additive,ham_concat}.yaml
}
fase4() {  # fusiones de scores en PlantVillage con batch 128 (~8.4 GB medidos; con 256 no cabe en 16 GB)
  # dot/additive/ham_concat con el mismo batch: referencia para no confundir el efecto con el batch
  # (compare.py --ref dot_b128).
  for v in ham_weighted ham_sum dot additive ham_concat; do
    scripts/run_seeds.sh "$SEEDS" $C/plantvillage_$v.yaml -- train.batch_size=128 name=plantvillage_${v}_b128
  done
}
fase4_food() {  # Food-101 (N=577): ~40 GB o más incluso con batch 128; sólo en GPUs grandes
  scripts/run_seeds.sh "$SEEDS" $C/food101_{ham_weighted,ham_sum}.yaml
}
fase5() {  # CIFAR-10 sin aumento de datos (régimen de las corridas de 2025) -> runs_noaug/
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_{dot,ham_concat,additive,ham_weighted,ham_sum}.yaml -- \
    data.augment=none train.output_dir=runs_noaug
}
fase6() {  # rama additive_free sola y escala normalizada (additive_free_scaled = /sqrt(d))
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_ham_{weighted,sum,weighted_learn}_scaled.yaml
  for v in additive_free ham_weighted_scaled ham_sum_scaled; do
    scripts/run_seeds.sh "$SEEDS" $C/plantvillage_$v.yaml -- train.batch_size=128 name=plantvillage_${v}_b128
  done
}
fase7() {  # más semillas de dot y ham_concat en CIFAR-10 (16 en total) para la tendencia de ham_concat
  scripts/run_seeds.sh "${SEEDS_EXTRA:-5 6 7 8 9 10 11 12 13 14 15}" $C/cifar10_{dot,ham_concat}.yaml -- \
    train.output_dir=runs_seeds  # aparte: las tablas de results/final siguen con 5 semillas
}
fase8() {  # caminos hacia un HAM funcional (INFORME_HAM.md, sección 8), CIFAR-10
  # A: controles de capacidad de ham_concat y variante con la rama por pares
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_{dot_wide,ham_dotdot,ham_concat_free}.yaml
  # B: mezclas de salidas (después del softmax) y scores estandarizados
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_ham_{wout,wout_free,wout_pairwise,wout_bdrop,gated,zscore}.yaml
  # C: heterogeneidad entre cabezas
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_ham_{heads,heads_free}.yaml
}
fase8_semillas() {  # controles de capacidad con las semillas 5-15, pareados con fase7 en runs_seeds/
  scripts/run_seeds.sh "${SEEDS_EXTRA:-5 6 7 8 9 10 11 12 13 14 15}" $C/cifar10_{dot_wide,ham_dotdot}.yaml -- \
    train.output_dir=runs_seeds
}
fase_salidas() {  # espejo de las fusiones de scores después del softmax (mezcla de salidas); no depende de fase8
  # CIFAR-10 (pareado con fase1/2/6): extremo λ=0, ponderada (λ 0.85/0.7/0.5), suma, λ aprendido y _scaled
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_additive_free.yaml \
    $C/cifar10_ham_{weighted,sum,weighted_learn}_out.yaml $C/cifar10_ham_{weighted,sum,weighted_learn}_out_scaled.yaml
  for lam in 0.5 0.7; do
    scripts/run_seeds.sh "$SEEDS" $C/cifar10_ham_weighted_out.yaml -- \
      "model.attention.fusion_kwargs.lambda_=$lam" "name=cifar10_ham_weighted_out_l${lam/./}"
  done
  # PlantVillage con batch 128 (pareado con fase4/6 y dot_b128)
  for v in ham_weighted_out ham_sum_out ham_weighted_out_scaled ham_sum_out_scaled; do
    scripts/run_seeds.sh "$SEEDS" $C/plantvillage_$v.yaml -- train.batch_size=128 name=plantvillage_${v}_b128
  done
}
fase_salidas_noaug() {  # espejo de fase5 (sin aumento) -> runs_noaug/
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_ham_{weighted,sum}_out.yaml -- data.augment=none train.output_dir=runs_noaug
}
ENRICH="dot_local dot_gate dot_global dot_local_gate dot_local_global"
fase10() {  # enriquecimientos de la salida de las cabezas (INFORME_HAM.md, camino G), CIFAR-10, 400 épocas
  # Referencias: dot (fase1) y dot_wide (fase8), mismo código y semillas (el camino sin enrich no cambió).
  scripts/run_seeds.sh "$SEEDS" $(for v in $ENRICH; do echo $C/cifar10_$v.yaml; done)
}
fase10_corta() {  # ¿alcanzan con 4 veces menos épocas? 100 épocas, con sus propias referencias -> runs_corta/
  scripts/run_seeds.sh "$SEEDS" $(for v in dot dot_wide $ENRICH; do echo $C/cifar10_$v.yaml; done) -- \
    train.epochs=100 train.warmup_epochs=3 train.output_dir=runs_corta
}
fase10_semillas() {  # las mejores de fase10 con las semillas 5-15, pareadas con fase7 (dot) y fase8_semillas (dot_wide)
  scripts/run_seeds.sh "${SEEDS_EXTRA:-5 6 7 8 9 10 11 12 13 14 15}" \
    $(for v in ${MEJORES:-dot_local dot_local_gate}; do echo $C/cifar10_$v.yaml; done) -- train.output_dir=runs_seeds
}
fase10_pv() {  # PlantVillage con batch 128, pareado con dot_b128 (fase4)
  for v in ${MEJORES_PV:-dot_local dot_local_gate}; do
    scripts/run_seeds.sh "$SEEDS" $C/plantvillage_$v.yaml -- train.batch_size=128 name=plantvillage_${v}_b128
  done
}
fase11() {  # ¿de dónde sale la ganancia de dot_local? (INFORME_HAM.md, 10.6), pareado con fase10
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_{dot_local_k1,dot_local_k5,ham_concat_local}.yaml
}
fase12() {  # dot_local_k5 con 16 semillas, kernel 7 y PlantVillage (INFORME_HAM.md, 10.9)
  scripts/run_seeds.sh "${SEEDS_EXTRA:-5 6 7 8 9 10 11 12 13 14 15}" $C/cifar10_dot_local_k5.yaml -- train.output_dir=runs_seeds
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_dot_local_k7.yaml
  scripts/run_seeds.sh "$SEEDS" $C/plantvillage_dot_local_k5.yaml -- train.batch_size=128 name=plantvillage_dot_local_k5_b128
}
fase13() {  # ham_concat + local 5x5 contra dot_local_k5 (INFORME_HAM.md, 10.11)
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_ham_concat_local_k5.yaml
}
fase14() {  # ¿la ventaja de ham_concat_local_k5 es capacidad? Control primero, luego 16 semillas (INFORME_HAM.md, 10.13)
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_dot_wide_local_k5.yaml
  scripts/run_seeds.sh "${SEEDS_EXTRA:-5 6 7 8 9 10 11 12 13 14 15}" $C/cifar10_ham_concat_local_k5.yaml -- train.output_dir=runs_seeds
}
fase14_extra() {  # el control de capacidad también con 16 semillas
  scripts/run_seeds.sh "${SEEDS_EXTRA:-5 6 7 8 9 10 11 12 13 14 15}" $C/cifar10_dot_wide_local_k5.yaml -- train.output_dir=runs_seeds
}
fase15() {  # variantes HAM con local 5x5, prioridad alta: pooling entre cabezas, control de proyección, sin pares
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_{ham_heads,ham_dotdot,additive}_local_k5.yaml
}
fase15b() {  # prioridad media: una segunda similitud por pares (mezcla de salidas, scores estandarizados) con local 5x5
  scripts/run_seeds.sh "$SEEDS" $C/cifar10_{ham_weighted_learn_out_scaled,ham_zscore}_local_k5.yaml
}
fase15_pv() {  # PlantVillage, donde la aditiva por pares gana: ¿se suma a local 5x5? Pareado con dot_b128
  for v in additive_free_local_k5 ham_sum_local_k5; do
    scripts/run_seeds.sh "$SEEDS" $C/plantvillage_$v.yaml -- train.batch_size=128 name=plantvillage_${v}_b128
  done
}
fase16() {  # ham_heads_local_k5 (criterio de promoción de 10.15) y additive_local_k5 (¿equivalente a dot_local_k5?) a 16 semillas
  scripts/run_seeds.sh "${SEEDS_EXTRA:-5 6 7 8 9 10 11 12 13 14 15}" $C/cifar10_{ham_heads,additive}_local_k5.yaml -- \
    train.output_dir=runs_seeds
}
fase17() {  # ham_concat_local_k5 - dot_wide_local_k5 (+0.49, p=0.17 con 16): n=48, contraste primario declarado antes
  scripts/run_seeds.sh "${SEEDS_EXTRA:-$(seq -s ' ' 16 47)}" $C/cifar10_{ham_concat_local_k5,dot_wide_local_k5}.yaml -- \
    train.output_dir=runs_seeds
}
fase17_pv() {  # la tríada y los controles de capacidad en PlantVillage, pareados con dot_b128
  for v in ham_concat_local_k5 dot_wide_local_k5 dot_wide; do
    scripts/run_seeds.sh "$SEEDS" $C/plantvillage_$v.yaml -- train.batch_size=128 name=plantvillage_${v}_b128
  done
}
# fase18 (revisión del artículo): semillas por fuera para tener resultados parciales de todas las variantes
CLAVE=(dot dot_wide dot_local_k5 dot_wide_local_k5 ham_concat_local_k5)  # variantes clave y sus controles de capacidad
fase18_mezcla() {  # ¿CutMix (Toledo 2026) o un aumento fuerte reducen la ganancia del término local?
  local cutmix=(train.mix=cutmix train.mix_alpha=1.0 train.mix_prob=0.6)  # el protocolo de Toledo 2026
  local fuerte=(data.augment=randaugment train.mix=cutmix_mixup train.mix_alpha=1.0 train.label_smoothing=0.1)
  for s in $SEEDS; do
    for v in "${CLAVE[@]}"; do
      scripts/run_seeds.sh "$s" $C/cifar10_$v.yaml -- name=cifar10_${v}_cutmix "${cutmix[@]}"
      scripts/run_seeds.sh "$s" $C/cifar10_$v.yaml -- name=cifar10_${v}_fuerte "${fuerte[@]}"
    done
  done
}
fase18_baselines() {  # baselines publicados a igual número de parámetros (119k-131k) y una ResNet de 121k
  for s in $SEEDS; do
    scripts/run_seeds.sh "$s" $C/cifar10_{localvit,spt_lsa,convit,conv_stem,resnet}.yaml
  done
}
fase18_datos() {  # CIFAR-100 (400 épocas) y SVHN (100 épocas, sin volteo)
  for s in $SEEDS; do
    scripts/run_seeds.sh "$s" $C/cifar100_{dot,dot_wide,dot_local_k5,dot_wide_local_k5,ham_concat_local_k5}.yaml \
      $C/svhn_{dot,dot_wide,dot_local_k5,dot_wide_local_k5,ham_concat_local_k5}.yaml
  done
}
fase18_fraccion() {  # 10, 25 y 50 % del train de CIFAR-10 (subconjunto fijo), 400 épocas
  for s in $SEEDS; do
    for f in 10 25 50; do
      for v in dot dot_local_k5 dot_wide_local_k5 ham_concat_local_k5; do
        scripts/run_seeds.sh "$s" $C/cifar10_$v.yaml -- data.train_fraction=0.$f name=cifar10_${v}_frac$f
      done
    done
  done
}
fase18_escala() {  # ancho D y profundidad L, 4 cabezas y MLP de 4D: ¿se reduce la ventaja del término local?
  for s in ${SEEDS_ESCALA:-0 1 2}; do
    for dl in 96:4 192:4 48:8 96:8 192:8; do
      local d=${dl%:*} l=${dl#*:}
      for v in dot dot_local_k5; do
        scripts/run_seeds.sh "$s" $C/cifar10_$v.yaml -- model.hidden_size=$d model.num_layers=$l \
          model.intermediate_size=$((4 * d)) name=cifar10_${v}_d${d}_l${l}
      done
    done
  done
}
fase18() { fase18_mezcla; fase18_baselines; fase18_datos; fase18_fraccion; fase18_escala; analizar; }
prelim() {  # métricas iniciales para redactar (GPU modesta); semilla por fuera: tabla completa cuanto antes
  local o=(-- train.epochs=20 train.warmup_epochs=2 data.augment=legacy train.output_dir=runs_prelim)
  for s in ${SEEDS_PRELIM:-0 1 2}; do
    scripts/run_seeds.sh "$s" \
      $C/cifar10_{dot,additive,ham_concat,ham_weighted,ham_sum,ham_weighted_learn}.yaml "${o[@]}"
    scripts/run_seeds.sh "$s" $C/plantvillage_{dot,additive,ham_concat}.yaml "${o[@]}"
  done
  reporte runs_prelim results/prelim
}
analizar() {
  for run in "${1:-runs}"/*/*/seed*; do
    [[ -f $run/test.json ]] || continue
    uv run python -m ham.analyze --run "$run" --skip-done
  done
}
reporte() {
  local runs=${1:-runs} dest=${2:-results/final}
  mkdir -p "$dest"
  analizar "$runs"
  uv run python scripts/aggregate.py --runs "$runs" --csv "$dest/results.csv" --latex "$dest/results.tex"
  uv run python scripts/plot_curves.py --runs "$runs" --out "$dest"
  uv run python scripts/compare.py --runs "$runs" --out "$dest"
  for run in "$runs"/*/*/seed0; do  # mapas por rama (rollout) y por capa de cada variante
    [[ -f $run/test.json ]] || continue
    local name=$(basename "$(dirname "$run")")
    uv run python -m ham.visualize --run "$run" --out "$dest/attention_${name}.png"
    uv run python -m ham.visualize --run "$run" --layers --out "$dest/attention_layers_${name}.png"
  done
  for d in "$runs"/*/; do  # todas las variantes sobre las mismas imágenes, con dot como referencia
    local ds=$(basename "$d")
    local ref="$d${ds}_dot/seed0"
    [[ -f $ref/predictions.npz ]] || continue
    local others=($(for r in "$d"*/seed0; do [[ $r != "$ref" && -f $r/predictions.npz ]] && echo "$r"; done))
    (( ${#others[@]} )) || continue
    uv run python -m ham.visualize --compare "$ref" "${others[@]}" --select disagree \
      --out "$dest/attention_compare_${ds}.png"
    uv run python -m ham.visualize --compare "$ref" "${others[@]}" \
      --out "$dest/attention_compare_random_${ds}.png"
  done
}
bench() {
  for cfg in $C/cifar10_{dot,additive,ham_concat,ham_weighted,ham_sum,ham_pairwise}.yaml \
             $C/plantvillage_{dot,ham_concat}.yaml $C/food101_{dot,ham_concat}.yaml; do
    scripts/run_seeds.sh "0" "$cfg" -- train.epochs=3 train.output_dir=runs_bench
  done
  uv run python - <<'PY'
import json, pathlib
for m in sorted(pathlib.Path("runs_bench").glob("*/*/seed0/metrics.json")):
    t = [r["time_s"] for r in json.loads(m.read_text())][1:]  # sin la 1a época (calentamiento)
    print(f"{m.parts[-4]:>13} {m.parts[-3]:<28} {sum(t)/len(t):7.1f} s/época")
PY
}
check() {
  uv run pytest -q
  uv run python - <<'PY'
import torch, pathlib
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO DISPONIBLE")
if torch.cuda.is_available():
    print(f"memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
root = pathlib.Path("data")
expected = {"cifar-10-batches-py": 8, "plantvillage": 54305, "food-101/images": 101001}
for d, n in expected.items():
    p = root / d
    found = sum(1 for f in p.rglob("*") if f.is_file()) if p.exists() else 0
    print(f"{'OK ' if found == n else 'MAL'} {d}: {found} archivos (esperados {n})")
PY
  [[ -z "$(git status --porcelain)" ]] && echo "OK  git limpio: $(git rev-parse --short HEAD)" \
    || echo "AVISO: hay cambios sin commit; provenance.json marcará dirty"
}

case "${1:-}" in
  check|bench|fase1|fase2|fase3|fase4|fase4_food|fase5|fase6|fase7|fase8|fase8_semillas|fase_salidas|fase_salidas_noaug|fase10|fase10_corta|fase10_semillas|fase10_pv|fase11|fase12|fase13|fase14|fase14_extra|fase15|fase15b|fase15_pv|fase16|fase17|fase17_pv|fase18|fase18_mezcla|fase18_baselines|fase18_datos|fase18_fraccion|fase18_escala|prelim) "$1" ;;
  fase9) fase_salidas ;;
  reporte) shift; reporte "$@" ;;
  analizar) shift; analizar "$@" ;;
  todo) fase1; fase5; fase2; fase3 ;;
  *) sed -n '2,21p' "$0"; exit 1 ;;
esac
