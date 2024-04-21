DEVICE=$1;
DATASET=$2;

ip=172.27.150.24;
export http_proxy=http://${ip}:7892;
export https_proxy=http://${ip}:7892;

bash scripts/train.sh $DEVICE $DATASET default;
bash scripts/train.sh $DEVICE $DATASET EntityMask;
bash scripts/train.sh $DEVICE $DATASET DataAug;
bash scripts/train.sh $DEVICE $DATASET Focal;
bash scripts/train.sh $DEVICE $DATASET RDrop;
bash scripts/train.sh $DEVICE $DATASET DFocal;
bash scripts/train.sh $DEVICE $DATASET PoE;