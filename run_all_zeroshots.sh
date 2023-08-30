
# Script arguments

LANGUAGE=${1}
MODEL_SIZE=${2}
PRETRAINED_LANGUAGE=${3}
SPECIFIC_SIZE=${4}

if [ -z "$LANGUAGE" ] || [ -z "$MODEL_SIZE" ]; then
  echo "Please, provide a language for finetune and model size. Current size supported is small"
  exit -1
fi

if [ -z "$PRETRAINED_LANGUAGE" ]; then
  echo "Skipping finetune from a pretrained model"
else 
  echo "Running finetune from model trained on $PRETRAINED_LANGUAGE"
fi

# Environment

PROJECT_DIR="./lang_transfer/"
T5X_DIR="./t5x"  # directory where the t5x is cloned.
BUCKET_NAME=${BUCKET_NAME:="lang_agnostic_europe"}

export PYTHONPATH="./"

# Experiments definition
MODEL_BASE_DIR="gs://${BUCKET_NAME}/models/finetune"
PRETRAINED_MODEL_CHECKPOINT="gs://${BUCKET_NAME}/models/pretrained/pretrained_${PRETRAINED_LANGUAGE}_${MODEL_SIZE}_6B/checkpoint_11445/"

echo "Running experiment with size 0 (Zero shot). Bucket is ${BUCKET_NAME}" ;

python3 ${T5X_DIR}/t5x/train.py \
    --gin_search_paths=${PROJECT_DIR} \
    --gin_file="lang_transfer/configs/runs/zeroshot.${MODEL_SIZE}.gin" \
    --gin.MODEL_DIR=\"${MODEL_BASE_DIR}/scratch_${LANGUAGE}_${MODEL_SIZE}_0M\" \
    --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.6M"\" \
    --gin.VAL_MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.validation"\"

if [ -n "$PRETRAINED_LANGUAGE" ]; then

  TRAIN_STEPS=$((10+11445))  # To account for pretraining steps

  python3 ${T5X_DIR}/t5x/train.py \
      --gin_search_paths=${PROJECT_DIR} \
      --gin_file="lang_transfer/configs/runs/zeroshot.${MODEL_SIZE}.gin" \
      --gin.MODEL_DIR=\"${MODEL_BASE_DIR}/${PRETRAINED_LANGUAGE}_${LANGUAGE}_${MODEL_SIZE}_0M\" \
      --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.6M"\" \
      --gin.VAL_MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.validation"\" \
      --gin.TRAIN_STEPS=${TRAIN_STEPS} \
      --gin.PRETRAINED_MODEL_PATH=\"${PRETRAINED_MODEL_CHECKPOINT}\"
fi

