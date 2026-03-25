#!/bin/bash

# ===== 用户可改参数 =====
ALLOWED_GPUS=(0 2 4 6)
REQUIRED_GPUS=4
MEM_THRESHOLD_MB=1000
UTIL_THRESHOLD=10
CHECK_INTERVAL=60

TRAIN_SCRIPT="/home/wangty/github/emu3/Emu3/scripts/tra_train.sh"

# ======================

find_free_gpus() {
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
    awk -F', ' -v mem_th="$MEM_THRESHOLD_MB" -v util_th="$UTIL_THRESHOLD" '
    {
        gpu_id=$1
        mem_used=$2
        util=$3
        if (mem_used < mem_th && util < util_th) {
            print gpu_id
        }
    }'
}

while true; do
    ALL_FREE=($(find_free_gpus))

    # 👉 只保留在 ALLOWED_GPUS 里的
    FREE_GPUS=()
    for gpu in "${ALL_FREE[@]}"; do
        for allowed in "${ALLOWED_GPUS[@]}"; do
            if [ "$gpu" -eq "$allowed" ]; then
                FREE_GPUS+=("$gpu")
            fi
        done
    done

    NUM_FREE=${#FREE_GPUS[@]}

    echo "[$(date '+%F %T')] Allowed GPUs: ${ALLOWED_GPUS[*]}"
    echo "[$(date '+%F %T')] Free GPUs (filtered): ${FREE_GPUS[*]} (count=${NUM_FREE})"

    if [ "$NUM_FREE" -ge "$REQUIRED_GPUS" ]; then
        SELECTED_GPUS=$(IFS=,; echo "${FREE_GPUS[*]:0:$REQUIRED_GPUS}")

        echo "[$(date '+%F %T')] Using GPUs: $SELECTED_GPUS"
        echo "[$(date '+%F %T')] Starting training..."

        export CUDA_VISIBLE_DEVICES="$SELECTED_GPUS"
        export NGPUS="$REQUIRED_GPUS"

        bash "$TRAIN_SCRIPT"
        exit $?
    fi

    echo "[$(date '+%F %T')] Not enough free GPUs, sleep ${CHECK_INTERVAL}s..."
    sleep "$CHECK_INTERVAL"
done