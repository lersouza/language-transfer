LANGUAGE=${1}
MODEL_SIZE=${2}

PROJECT_DIR="./lang_transfer/"
T5X_DIR="./t5x"  # directory where the t5x is cloned.
FINTETUNE_SIZES="1000000 10000000 100000000 1000000000"


export PYTHONPATH="./"

if [ -z "$LANGUAGE" ] || [ -z "$MODEL_SIZE" ]; then
  echo "Please, provide a language for finetune and model size. Current size supported is small"
  exit -1
fi

for size in $FINTETUNE_SIZES 
do 
  echo "Processing size $size"
  
  MODEL_DIR="gs://lang_agnostic/models/finetune/scratch_${LANGUAGE}_${MODEL_SIZE}_${size}/"

  python3 ${T5X_DIR}/t5x/train.py \
    --gin_search_paths=${PROJECT_DIR} \
    --gin_file="lang_transfer/configs/runs/finetune.${MODEL_SIZE}.gin" \
    --gin.MODEL_DIR=\"${MODEL_DIR}\" \
    --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.finetune.${LANGUAGE}.${MODEL_SIZE}.${size}"\"
done


