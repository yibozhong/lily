export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export BASE_MODEL=tiiuae/falcon-mamba-7b
CUDA_VISIBLE_DEVICES=3 python finetune_mamba.py \
  --base_model 'tiiuae/falcon-mamba-7b' \
  --data_path 'ft-training_set/commonsense_170k.json' \
  --output_dir './trained_models/mamba-lily' \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 1 \
  --learning_rate 5e-4 \
  --ne_1 4 \
  --ne_2 4 \
  --lily_r 40 \
  --lora_r 2 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lily \