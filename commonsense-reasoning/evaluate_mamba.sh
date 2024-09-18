export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export BASE_MODEL=tiiuae/falcon-mamba-7b

for dataset in "openbookqa" "boolq" "piqa" "social_i_qa" "hellaswag" "winogrande" "ARC-Challenge" "ARC-Easy"; do
    CUDA_VISIBLE_DEVICES=2 python evaluate_mamba.py \
        --model 'Falcon-Mamba' \
        --batch_size 1 \
        --adapter 'Lily' \
        --dataset "$dataset" \
        --base_model 'tiiuae/falcon-mamba-7b' \
        --lora_weights 'trained_models/mamba-lily'
done