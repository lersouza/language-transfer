# Script arguments

PRETRAINED_LANGUAGE=${1}

if [ -z "$PRETRAINED_LANGUAGE" ]; then
  echo "Please, provide a pretraining language for finetune"
fi

# Environment

PROJECT_DIR="./lang_transfer/"
T5X_DIR="./t5x"  # directory where the t5x is cloned.
BUCKET_NAME=${BUCKET_NAME:="lang_agnostic_europe"}

export PYTHONPATH="./"

# Experiments definition
MODEL_BASE_DIR="gs://${BUCKET_NAME}/models/tasks"
PRETRAINED_MODEL_CHECKPOINT="gs://${BUCKET_NAME}/models/pretrained/pretrained_${PRETRAINED_LANGUAGE}_${MODEL_SIZE}_6B/checkpoint_11445/"

python3 ${T5X_DIR}/t5x/train.py \
    --gin_search_paths=${PROJECT_DIR} \
    --gin_file="lang_transfer/configs/runs/train_scratch.${MODEL_SIZE}.gin" \
    --gin.MODEL_DIR=\"${MODEL_BASE_DIR}/assin2_from_${PRETRAINED_MODEL_CHECKPOINT}\" \
    --gin.MIXTURE_OR_TASK_NAME=\""assin2"\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"${PRETRAINED_MODEL_CHECKPOINT}\"
