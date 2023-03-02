LANGUAGE=${1}
PRETRAIN_SIZE=${2}

PROJECT_DIR="./lang_transfer/"
T5X_DIR="./t5x"  # directory where the t5x is cloned.
MODEL_DIR="gs://lang_agnostic/models/$LANGUAGE\_$PRETRAIN_SIZE/"

export PYTHONPATH="./"

if [ -z "$LANGUAGE" ] || [ -z "$PRETRAIN_SIZE" ]; then
  echo "Please, provide a language for pretraining and corpus size. Current size supported is small"
  exit -1
fi


python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="lang_transfer/configs/runs/pretrain.$PRETRAIN_SIZE.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.pretrain.$LANGUAGE.$PRETRAIN_SIZE"\"

