#!/usr/bin/env bash
# command: ./experiment.sh EXP_NAME [EPOCHS] [LR] [SIGMA]

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 EXP_NAME [LR] [FILENAME] [SCHEDULE] [IS_A100] [PARAMETER]"
  exit 1
fi

# exp. parameters 
EXP_NAME=$1
LR=${2:-0.005}
PREPROCESS=$3
FILENAME=$4
SCHEDULE=$5
IS_A100=$6
# PARAMETER=$6

LR_FT=$(python3 -c "print(float('$LR')/2)")

# dir. setting
LOG_DIR="log"
CKPT_DIR="ckpt"
OUT_DIR="output"
SCHE_DIR="schedule"

mkdir -p "$LOG_DIR/$EXP_NAME/$FILENAME" "$CKPT_DIR/$EXP_NAME/$FILENAME" "$OUT_DIR/$EXP_NAME/$FILENAME"

# 1) training
python3 train.py \
  --lr "$LR" \
  --desc "$EXP_NAME" \
  --log_path "$LOG_DIR/$EXP_NAME/$FILENAME/sdf_model_${EXP_NAME}_${FILENAME}.txt" \
  --ckpt_path "$CKPT_DIR/$EXP_NAME/$FILENAME/sdf_model_${EXP_NAME}_${FILENAME}" \
  --preprocess "$PREPROCESS" \
  --file_name "$FILENAME" \
  --schedule_path "$SCHE_DIR/${SCHEDULE}.json" \
  --is_a100 "$IS_A100" 
  # --para "$PARAMETER"
echo -e "\033[32m[1/2] Finish Training\033[0m"

# 2) inference
# epochs=("epoch1000" "epoch1500" "final")
# epochs=("finalitri")
# epochs=("finetune_final")
# epochs=("final" "finetune_final")
# epochs=("final" "epoch50" "epoch100" "epoch150")
epochs=("final")

for epoch in "${epochs[@]}"; do
  python3 inference.py \
    --res 350 \
    --ckpt_path "$CKPT_DIR/$EXP_NAME/$FILENAME/sdf_model_${EXP_NAME}_${FILENAME}_${epoch}.pt" \
    --output_mesh "$OUT_DIR/$EXP_NAME/$FILENAME/sdf_model_${EXP_NAME}_${FILENAME}_${epoch}" \
    --preprocess "$PREPROCESS" \
    --file_name "$FILENAME"
    # --plan_y 0.5 \
    # --start_xz 0.1 0.1 \
    # --goal_xz 0.9 0.9 \
    # --rrt_step 4 --rrt_iters 8000 \
    # --goal_sample_rate 0.01 \
    # --rewire_radius 30 \
    # --free_thresh 0.0 \
    # --clearance 0.02 \
    # --target_iso 0.05 \
    # --w_iso 2000.0 \
    # --w_margin 5.0
  echo -e "\033[32m[2/2] Finish Inferencing ${FILENAME} ${epoch}\033[0m"
done