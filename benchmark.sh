#!/usr/bin/env bash
# =====================================================================
# benchmark.sh — Comparativa de tiempos GPU vs CPU del MLP (main4).
#
# Ejecuta una matriz de experimentos:
#   4 arquitecturas (0 a 3 capas ocultas) x 2 devices (cuda/cpu)
#   x 2 dtypes (f32/f64) = 16 runs
#
# Para cada experimento mide:
#   - tiempo total de entrenamiento y tiempo medio por epoca
#   - precision en train y en test
#
# Toda la salida se escribe EN STREAMING a comparativa.txt (via tee):
# puedes verla en pantalla mientras se genera, y al final queda la
# tabla resumen con el factor de aceleracion GPU vs CPU.
# Ademas se genera comparativa.csv con las mismas metricas (para graficas).
#
# Uso:
#   ./benchmark.sh                  # compila y lanza la comparativa completa
#   ./benchmark.sh --no-build       # sin recompilar (usa el binario actual)
#
# Variables de entorno (opcionales):
#   EPOCHS=5 ./benchmark.sh         # menos epocas (pasada rapida)
#   LIMIT=10000 ./benchmark.sh      # limitar muestras de entrenamiento
#   OUTFILE=otro.txt ./benchmark.sh # cambiar el archivo de salida
#   ARCHS_LIST="784,64,10 784,128,64,32,10" ./benchmark.sh
#                                   # arquitecturas propias (cualquier
#                                   # profundidad/anchura, separadas por
#                                   # espacios; primera capa 784, ultima 10)
#
# Duracion estimada con los valores por defecto (15 epocas, dataset
# completo): ~35-40 min, dominado por los runs de CPU (f64).
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-15}
LIMIT=${LIMIT:--1}
OUTFILE=${OUTFILE:-comparativa.txt}

BUILD=1
for arg in "$@"; do
    case "$arg" in
        --no-build) BUILD=0 ;;
        *) echo "[ERROR] Argumento desconocido: $arg"; exit 1 ;;
    esac
done

# Lista de arquitecturas (cualquier profundidad/anchura valida; primera capa
# 784, ultima 10). Override: ARCHS_LIST="784,64,10 784,128,64,32,10" ./benchmark.sh
if [[ -n "${ARCHS_LIST:-}" ]]; then
    read -ra ARCHS <<< "$ARCHS_LIST"
else
    # 0, 1, 2 y 3 capas ocultas (mismo patron de anchura)
    ARCHS=("784,10" "784,128,10" "784,128,64,10" "784,128,64,32,10")
fi
DEVICES=("cuda" "cpu")
DTYPES=("f32" "f64")

# ---------------- Compilacion ----------------
if [[ $BUILD -eq 1 ]]; then
    echo "[BUILD] Compilando main4 (nvcc)..."
    nvcc -std=c++17 -O2 main4.cpp \
         lib/tensor.cpp lib/ops.cpp lib/linear.cpp lib/loss.cpp \
         lib/kernel_operations.cu lib/ops.cu lib/loss.cu \
         -o train_mnist.out -lcudart -lcurand
fi

if [[ ! -x ./train_mnist.out ]]; then
    echo "[ERROR] No existe ./train_mnist.out (ejecuta sin --no-build)"
    exit 1
fi

# ---------------- Cabecera ----------------
: > "$OUTFILE"
CSVFILE="${OUTFILE%.txt}.csv"
echo "arquitectura,device,dtype,tiempo_s,s_epoca,acc_train,acc_test" > "$CSVFILE"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "sin GPU")
FECHA=$(date '+%Y-%m-%d %H:%M:%S')

{
echo "======================================================================"
echo " COMPARATIVA GPU vs CPU - MLP en MNIST (main4)"
echo " Fecha:          $FECHA"
echo " GPU:            $GPU_NAME"
echo " Arquitecturas:  ${ARCHS[*]}"
echo " Dtypes:         ${DTYPES[*]}"
echo " Epocas:         $EPOCHS"
if [[ $LIMIT -lt 0 ]]; then
    echo " Muestras train: todas (60000)"
else
    echo " Muestras train: $LIMIT (limitadas)"
fi
echo "======================================================================"
echo ""
} | tee -a "$OUTFILE"

# ---------------- Matriz de experimentos ----------------
declare -A RES_TIME RES_EPOCH RES_ACC_TR RES_ACC_TE
N=$(( ${#ARCHS[@]} * ${#DEVICES[@]} * ${#DTYPES[@]} ))
i=0

for ARCH in "${ARCHS[@]}"; do
    for DEV in "${DEVICES[@]}"; do
        for DT in "${DTYPES[@]}"; do
            i=$((i + 1))

            {
            echo "----------------------------------------------------------------------"
            echo " [${i}/${N}] arquitectura=${ARCH} | device=${DEV} | dtype=${DT} | epocas=${EPOCHS}"
            echo "----------------------------------------------------------------------"
            } | tee -a "$OUTFILE"

            RUNLOG=$(mktemp)
            stdbuf -oL -eL ./train_mnist.out \
                --device "$DEV" --dtype "$DT" --layers "$ARCH" \
                --epochs "$EPOCHS" --limit "$LIMIT" \
                2>&1 | tee -a "$OUTFILE" "$RUNLOG"

            # ---- Extraer metricas de la salida ----
            TIME=$(grep -oP 'Tiempo total de entrenamiento: \K[0-9.]+' "$RUNLOG" | tail -1 || true)
            ACC_TR=$(grep -oP 'Precision en Entrenamiento: \K[0-9.]+' "$RUNLOG" | tail -1 || true)
            ACC_TE=$(grep -oP 'Precision en Test Set:\s+\K[0-9.]+' "$RUNLOG" | tail -1 || true)
            AVG_EP=$(grep -oP 'Epoca \[[^]]+\].*\|\s+\K[0-9.]+(?=s$)' "$RUNLOG" \
                     | awk '{s+=$1; n++} END {if (n>0) printf "%.2f", s/n; else print "NA"}' || true)

            TIME=${TIME:-NA}; ACC_TR=${ACC_TR:-NA}; ACC_TE=${ACC_TE:-NA}; AVG_EP=${AVG_EP:-NA}

            key="$ARCH|$DEV|$DT"
            RES_TIME[$key]=$TIME
            RES_EPOCH[$key]=$AVG_EP
            RES_ACC_TR[$key]=$ACC_TR
            RES_ACC_TE[$key]=$ACC_TE

            # CSV para graficas (arquitectura entre comillas por las comas)
            echo "\"$ARCH\",$DEV,$DT,$TIME,$AVG_EP,$ACC_TR,$ACC_TE" >> "$CSVFILE"

            {
            echo ""
            printf ">>> RESULTADO [%s | %s | %s]: tiempo=%ss | s/epoca=%s | acc_train=%s%% | acc_test=%s%%\n" \
                   "$ARCH" "$DEV" "$DT" "$TIME" "$AVG_EP" "$ACC_TR" "$ACC_TE"
            echo ""
            } | tee -a "$OUTFILE"

            rm -f "$RUNLOG"
        done
    done
done

# ---------------- Tabla resumen ----------------
{
echo ""
echo "======================================================================"
echo " TABLA RESUMEN"
echo "======================================================================"
printf "%-24s %-7s %-6s %10s %9s %10s %10s\n" "Arquitectura" "Device" "Dtype" "Tiempo(s)" "s/epoca" "AccTrain" "AccTest"
echo "----------------------------------------------------------------------"
for ARCH in "${ARCHS[@]}"; do
    for DEV in "${DEVICES[@]}"; do
        for DT in "${DTYPES[@]}"; do
            key="$ARCH|$DEV|$DT"
            printf "%-24s %-7s %-6s %10s %9s %10s %10s\n" \
                "$ARCH" "$DEV" "$DT" \
                "${RES_TIME[$key]:-NA}" "${RES_EPOCH[$key]:-NA}" \
                "${RES_ACC_TR[$key]:-NA}%" "${RES_ACC_TE[$key]:-NA}%"
        done
    done
done
echo ""
echo " Factor de aceleracion GPU vs CPU (tiempo CPU / tiempo GPU):"
for ARCH in "${ARCHS[@]}"; do
    for DT in "${DTYPES[@]}"; do
        kc="$ARCH|cuda|$DT"; kp="$ARCH|cpu|$DT"
        tc=${RES_TIME[$kc]:-}; tp=${RES_TIME[$kp]:-}
        if [[ -n "$tc" && -n "$tp" && "$tc" != "NA" && "$tp" != "NA" ]]; then
            SP=$(awk -v a="$tp" -v b="$tc" 'BEGIN { if (b > 0) printf "%.1fx", a/b; else print "NA" }')
            printf "   %-24s %-5s : %s\n" "$ARCH" "$DT" "$SP"
        fi
    done
done
echo "======================================================================"
echo " Resultados guardados en: $OUTFILE"
echo "                          $CSVFILE"
echo "======================================================================"
} | tee -a "$OUTFILE"
