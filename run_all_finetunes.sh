# Script arguments

LANGUAGE=${1}
MODEL_SIZE=${2}
PRETRAINED_LANGUAGE=${3}
SKIP_SCRATCH=${4}
SPECIFIC_SIZE=${5}

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
BUCKET_NAME=${BUCKET_NAME:="lang_agnostic"}

export PYTHONPATH="./"

# Experiments definition
FINETUNE_SIZES=("6M" "19M" "60M" "189M" "600M" "6B")
FINETUNE_EPOCH_STEPS=(12 37 115 361 1145 11445)  # number of steps to form an epoch
EPOCHS=(10 10 10 10 10 3)  # number of steps to form an epoch
WARMUP_STEPS=(0 0 30 100 300 3000)
MODEL_BASE_DIR="gs://${BUCKET_NAME}/models/finetune"
PRETRAINED_MODEL_CHECKPOINT="gs://${BUCKET_NAME}/models/pretrained/pretrained_${PRETRAINED_LANGUAGE}_${MODEL_SIZE}_6B/checkpoint_11445/"

RUNS=${#FINETUNE_SIZES[@]}



for (( i=0; i<$RUNS; i++ )); do
    DATA_SIZE=${FINETUNE_SIZES[$i]}
    EPOCH_STEPS=${FINETUNE_EPOCH_STEPS[$i]}
    EPOCHS_TO_TRAIN=${EPOCHS[$i]}
    WARMUP=${WARMUP_STEPS[$i]}

    TRAIN_STEPS=$((EPOCH_STEPS*EPOCHS_TO_TRAIN))
    EVAL_PERIOD=$((EPOCH_STEPS))

    if [ ! -z "$SPECIFIC_SIZE" ] && [ "$SPECIFIC_SIZE" != "$DATA_SIZE" ]; then
      echo "Skipping size $DATA_SIZE"
      continue
    fi
    
    echo "Running experiment with size ${DATA_SIZE}, # of train steps ${TRAIN_STEPS}, #warmup ${WARMUP}. Bucket is ${BUCKET_NAME}" ;

    if [ "$SKIP_SCRATCH" -ne "1" ]; then

      python3 ${T5X_DIR}/t5x/train.py \
          --gin_search_paths=${PROJECT_DIR} \
          --gin_file="lang_transfer/configs/runs/train_scratch.${MODEL_SIZE}.gin" \
          --gin.MODEL_DIR=\"${MODEL_BASE_DIR}/scratch_${LANGUAGE}_${MODEL_SIZE}_${DATA_SIZE}\" \
          --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.${DATA_SIZE}"\" \
          --gin.VAL_MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.validation"\" \
          --gin.TRAIN_STEPS=${TRAIN_STEPS} \
          --gin.EVAL_PERIOD=${EVAL_PERIOD} \
          --gin.WARMUP_STEPS=${WARMUP}
    fi

    if [ -n "$PRETRAINED_LANGUAGE" ]; then

      TRAIN_STEPS=$((TRAIN_STEPS+11445))  # To account for pretraining steps

      python3 ${T5X_DIR}/t5x/train.py \
          --gin_search_paths=${PROJECT_DIR} \
          --gin_file="lang_transfer/configs/runs/finetune.${MODEL_SIZE}.gin" \
          --gin.MODEL_DIR=\"${MODEL_BASE_DIR}/${PRETRAINED_LANGUAGE}_${LANGUAGE}_${MODEL_SIZE}_${DATA_SIZE}\" \
          --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.${DATA_SIZE}"\" \
          --gin.VAL_MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.validation"\" \
          --gin.TRAIN_STEPS=${TRAIN_STEPS} \
          --gin.EVAL_PERIOD=${EVAL_PERIOD} \
          --gin.WARMUP_STEPS=0 \
          --gin.PRETRAINED_MODEL_PATH=\"${PRETRAINED_MODEL_CHECKPOINT}\"
    fi
done
