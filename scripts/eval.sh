# bash scripts/eval.sh 0 tacred $TRAIN_MODE test

CUDA_DEVICE=$1;
export DATASET=$2;
export TRAIN_MODE=$3;
export VALIDATION_DATA_PATH=$4;
export SEED=42;
export PRED_ROOT=data/predictions;
# example of LUKE
allennlp evaluate results/relation_classification/${DATASET}/luke-base-${TRAIN_MODE} ${VALIDATION_DATA_PATH} \
  --include-package examples --output-file results/relation_classification/${DATASET}/luke-base-${TRAIN_MODE}/metrics_test.json --cuda-device ${CUDA_DEVICE}
