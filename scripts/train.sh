# set_proxy first;
# bash scripts/train.sh 0 tacred default
export CUDA_DEVICE=$1;
export DATASET=$2;
export TRAIN_DATA_PATH="data/$DATASET/train4debias.json";
export VALIDATION_DATA_PATH="data/$DATASET/dev.json";
export TRAIN_MODE=$3;
export SEED=42;

rm -rf results/relation_classification/${DATASET}/luke-base-$TRAIN_MODE;

# train LUKE
export TRANSFORMERS_MODEL_NAME="studio-ousia/luke-base";
allennlp train examples/relation_classification/configs/transformers_luke_with_entity_aware_attention.jsonnet \
  -s results/relation_classification/${DATASET}/luke-base-$TRAIN_MODE \
  --include-package examples -o '{"trainer.use_amp": true}'