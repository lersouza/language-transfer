#!/bin/bash

# Script arguments

LANGUAGE=${1}
MODEL_SIZE=${2}
PRETRAINED_LANGUAGE=${3}

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

ROJECT_DIR="./lang_transfer/"
T5X_DIR="./t5x"  # directory where the t5x is cloned.
BUCKET_NAME=${BUCKET_NAME:="lang_agnostic"}

export PYTHONPATH="./"

# Experiments definition
MODEL_BASE_DIR="gs://${BUCKET_NAME}/models/finetune"
SCRATCH_MODEL_CHECKPOINT="gs://${BUCKET_NAME}/models/pretrained/pretrained_${PRETRAINED_LANGUAGE}_${MODEL_SIZE}_6B/checkpoint_0/"
PRETRAINED_MODEL_CHECKPOINT="gs://${BUCKET_NAME}/models/pretrained/pretrained_${PRETRAINED_LANGUAGE}_${MODEL_SIZE}_6B/checkpoint_11445/"


python3 ${T5X_DIR}/t5x/eval.py \
    --gin_search_paths=${PROJECT_DIR} \
    --gin_file="lang_transfer/configs/runs/eval.${MODEL_SIZE}.gin" \
    --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.validation"\" \
    --gin.CHECKPOINT_PATH=\"${SCRATCH_MODEL_CHECKPOINT}\" \
    --gin.EVAL_OUTPUT_DIR=\"${MODEL_BASE_DIR}/scratch_${LANGUAGE}_${MODEL_SIZE}_0M\"

if [ -n "$PRETRAINED_LANGUAGE" ]; then
    python3 ${T5X_DIR}/t5x/eval.py \
        --gin_search_paths=${PROJECT_DIR} \
        --gin_file="lang_transfer/configs/runs/eval.${MODEL_SIZE}.gin" \
        --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.validation"\" \
        --gin.CHECKPOINT_PATH=\"${SCRATCH_MODEL_CHECKPOINT}\" \
        --gin.EVAL_OUTPUT_DIR=\"${MODEL_BASE_DIR}/${PRETRAINED_LANGUAGE}_${LANGUAGE}_${MODEL_SIZE}_0M\"
fi