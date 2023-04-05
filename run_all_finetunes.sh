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

PROJECT_DIR="./lang_transfer/"
T5X_DIR="./t5x"  # directory where the t5x is cloned.

export PYTHONPATH="./"

# Experiments definition
FINETUNE_SIZES=(6000000 60000000 600000000 6000000000)
FINETUNE_EPOCH_STEPS=(11 114 1144 11444)  # number of steps to form an epoch
WARMUP_STEPS=(0 30 300 3000)
MODEL_BASE_DIR="gs://lang_agnostic/models"
PRETRAINED_MODEL_CHECKPOINT="gs://lang_agnostic/models/scratch_${PRETRAINED_LANGUAGE}_${MODEL_SIZE}_6000000000/checkpoint_10682/"

RUNS=${#FINETUNE_SIZES[@]}



for (( i=0; i<$RUNS; i++ )); do
    DATA_SIZE=${FINETUNE_SIZES[$i]}
    EPOCH_STEPS=${FINETUNE_EPOCH_STEPS[$i]}
    WARMUP=${WARMUP_STEPS[$i]}

    TRAIN_STEPS=$((EPOCH_STEPS*10))
    EVAL_PERIOD=$((EPOCH_STEPS))
    
    echo "Running experiment with size ${DATA_SIZE}, # of train steps ${TRAIN_STEPS}, #warmup ${WARMUP}" ;

    python3 ${T5X_DIR}/t5x/train.py \
        --gin_search_paths=${PROJECT_DIR} \
        --gin_file="lang_transfer/configs/runs/train_scratch.${MODEL_SIZE}.gin" \
        --gin.MODEL_DIR=\"${MODEL_BASE_DIR}/scratch_${LANGUAGE}_${MODEL_SIZE}_${DATA_SIZE}\" \
        --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.finetune.${LANGUAGE}.${MODEL_SIZE}.${DATA_SIZE}"\" \
        --gin.TRAIN_STEPS=${TRAIN_STEPS} \
        --gin.EVAL_PERIOD=${EVAL_PERIOD} \
        --gin.WARMUP_STEPS=${WARMUP}

    if [ -n "$PRETRAINED_LANGUAGE" ]; then

      TRAIN_STEPS=$((TRAIN_STEPS+10682))  # To account for pretraining steps

      python3 ${T5X_DIR}/t5x/train.py \
          --gin_search_paths=${PROJECT_DIR} \
          --gin_file="lang_transfer/configs/runs/finetune.${MODEL_SIZE}.gin" \
          --gin.MODEL_DIR=\"${MODEL_BASE_DIR}/en_${LANGUAGE}_${MODEL_SIZE}_${DATA_SIZE}\" \
          --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.finetune.${LANGUAGE}.${MODEL_SIZE}.${DATA_SIZE}"\" \
          --gin.TRAIN_STEPS=${TRAIN_STEPS} \
          --gin.EVAL_PERIOD=${EVAL_PERIOD} \
          --gin.WARMUP_STEPS=0 \
          --gin.PRETRAINED_MODEL_PATH=\"${PRETRAINED_MODEL_CHECKPOINT}\"
    fi
done
