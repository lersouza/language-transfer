LANGUAGE=${1}

PROJECT_DIR="./lang_transfer/"
T5X_DIR="./t5x"  # directory where the t5x is cloned.

DATA_SIZE="6B"
MODEL_DIR="gs://lang_agnostic/models/pretrained/pretrained_${LANGUAGE}_small_${DATA_SIZE}/"

export PYTHONPATH="./"

if [ -z "$LANGUAGE" ]; then
  echo "Please, provide a language for pretraining."
  exit -1
fi


python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="lang_transfer/configs/runs/train_scratch.small.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.${DATA_SIZE}"\" \
  --gin.VAL_MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.validation"\" \
  --gin.TRAIN_STEPS=11445 \
  --gin.EVAL_PERIOD=1000 \
  --gin.WARMUP_STEPS=3000
