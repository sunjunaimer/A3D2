set -e
modality=$1
anchor=$2
run_idx=$3
gpu=$4
source=$5
target=$6
log_dir=$7
csv_name=$8
lr=$9
weights=${10}
change_weight1=${11}
change_weight2=${12}
model=${13}

current_timestamp=$(date +"%m%d%H%M")

for i in `seq 1 1 100`;
do

# bash scripts/xx_dlp15.sh AVL 1 1 meld_multimodal msp_multimodal MELD MSP .logs/acm/p1/0.1/without_bn  #emotion_light_cdan

cmd="CUDA_VISIBLE_DEVICES=$gpu python3 train_msa_po.py --model=$model
--gpu_ids=0 --modality=$modality --anchor=$anchor --source=$source --target=$target
--log_dir=$log_dir --checkpoints_dir=./checkpoints --print_freq=10   --csv_name=$csv_name
--input_dim_a=768 --embd_size_a=256 --embd_method_a=maxpool
--input_dim_v=768 --embd_size_v=256  --embd_method_v=maxpool
--input_dim_l=768 --embd_size_l=256
--output_dim=4 --cls_layers=128,128 --dropout_rate=0.3
--niter=4 --niter_decay=4 --in_mem --beta1=0.9
--batch_size=48 --lr=1e-3 --run_idx=$run_idx
--name=$current_timestamp --suffix={modality}_{source}_2_{target}
--has_test --cvNo=$i
--weights $weights  
--change_weight1=$change_weight1
--change_weight2=$change_weight2
--disc_test=True
--serial_batches=False
--cof_weight=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done