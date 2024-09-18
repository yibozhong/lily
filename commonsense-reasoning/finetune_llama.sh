export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export BASE_MODEL=meta-llama/Meta-Llama-3-8B
CUDA_VISIBLE_DEVICES=3 python finetune_llama.py \
  --base_model 'meta-llama/Meta-Llama-3-8B' \
  --data_path 'ft-training_set/commonsense_170k.json' \
  --output_dir './trained_models/llama-lily' \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --ne_1 4 \
  --ne_2 4 \
  --lily_r 16 \
  --lora_r 16 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lily \