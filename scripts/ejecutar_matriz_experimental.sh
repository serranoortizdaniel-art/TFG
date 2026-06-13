#!/usr/bin/env bash
# ============================================================
# Ejecucion reanudable de la matriz experimental final del TFG
#
# Uso recomendado:
#   bash scripts/ejecutar_matriz_experimental.sh gpu
#
# Otros usos:
#   bash scripts/ejecutar_matriz_experimental.sh cpu
#
# La ejecucion es reanudable: si se interrumpe, al relanzar el mismo
# comando se omiten las ejecuciones que ya esten completadas.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROYECTO="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="${PFG_FLOWER_VENV:-$HOME/.venvs/pfg-flower}"

PY="$VENV/bin/python3"
MODO="${1:-gpu}"

if [ "$MODO" != "gpu" ] && [ "$MODO" != "cpu" ]; then
  echo "ERROR: modo no valido: $MODO"
  echo "Uso: bash scripts/ejecutar_matriz_experimental.sh [gpu|cpu]"
  exit 1
fi

[ -x "$PY" ] || {
  echo "ERROR: no encuentro Python en $VENV"
  echo "Define PFG_FLOWER_VENV o crea el entorno en ~/.venvs/pfg-flower."
  exit 1
}

cd "$PROYECTO" || {
  echo "ERROR: no encuentro el proyecto en $PROYECTO"
  echo "Edita la variable PROYECTO al principio del script."
  exit 1
}

mkdir -p results
PID_FILE="results/matriz_experimental.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "ERROR: ya hay una matriz experimental en ejecucion."
  echo "PID activo: $(cat "$PID_FILE")"
  echo "Log: ls -t results/matriz_experimental_*.log | head -1"
  exit 1
fi

export PYTHONUNBUFFERED=1
# No deduplicar logs de Ray durante tiradas largas: las repeticiones ayudan a
# diagnosticar cuellos de botella o avisos intermitentes entre ejecuciones.
export RAY_DEDUP_LOGS=0

# Los factores cientificos de la matriz viven en run_all.py; este script solo
# fija el perfil de ejecucion validado para la maquina de desarrollo.
ARGS=(--dataset fashion_mnist)

if [ "$MODO" = "gpu" ]; then
  # Perfil validado para RTX 3050/3050 Ti Laptop de 4 GB:
  # un unico actor Ray usa la GPU para evitar saturar la VRAM.
  ARGS+=(--device cuda --client-device cuda --client-gpus 1 --client-cpus 4 --ray-cpus 4)
else
  # Perfil CPU para Ryzen 7 H-series: hasta 4 clientes virtuales en paralelo.
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  ARGS+=(--device cpu --client-device cpu --ray-cpus 8 --client-cpus 2)
fi

LOG="results/matriz_experimental_${MODO}.log"

echo "=== Matriz experimental final del TFG ===" | tee -a "$LOG"
echo "Modo de ejecucion: $MODO"                  | tee -a "$LOG"
echo "Proyecto: $PROYECTO"                       | tee -a "$LOG"
echo "Python: $PY"                               | tee -a "$LOG"
echo ""                                         | tee -a "$LOG"
echo "=== Resumen del plan ==="                  | tee -a "$LOG"
PLAN_OUTPUT="$("$PY" run_all.py --dry-run "${ARGS[@]}" 2>&1)"
echo "${PLAN_OUTPUT%%$'\n'*}"                    | tee -a "$LOG"
echo ""                                         | tee -a "$LOG"
echo "Inicio: $(date)"                           | tee -a "$LOG"

nohup "$PY" run_all.py "${ARGS[@]}" >> "$LOG" 2>&1 &
echo $! > "$PID_FILE"

echo ""
echo "Matriz lanzada en segundo plano."
echo "PID: $(cat "$PID_FILE")"
echo "Log: $PROYECTO/$LOG"
echo ""
echo "Ver progreso:"
echo "  tail -f \"$PROYECTO/$LOG\""
echo ""
echo "Parar la ejecucion:"
echo "  kill \$(cat \"$PROYECTO/$PID_FILE\")"
echo ""
echo "Reanudar despues de parar o reiniciar:"
echo "  bash scripts/ejecutar_matriz_experimental.sh $MODO"
