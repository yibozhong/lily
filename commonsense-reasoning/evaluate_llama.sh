export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export BASE_MODEL=meta-llama/Meta-Llama-3-8B

for dataset in "openbookqa" "boolq" "piqa" "social_i_qa" "hellaswag" "winogrande" "ARC-Challenge" "ARC-Easy"; do
    CUDA_VISIBLE_DEVICES=3 python evaluate_llama.py \
        --model 'LLaMA-7B' \
        --batch_size 1 \
        --adapter 'Lily' \
        --dataset "$dataset" \
        --base_model 'meta-llama/Meta-Llama-3-8B' \
        --lora_weights 'trained_models/llama-lily'
done