
#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

for subject in "bear_plushie" "grey_sloth_plushie" "monster_toy" "rc_car" "red_cartoon" "robot_toy" "wolf_plushie"; do
  export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
  export INSTANCE_DIR=dataset/$subject
  export OUTPUT_DIR=lora-trained-xl-$subject
  export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
  #! vanilla lora
  CUDA_VISIBLE_DEVICES=3 python train_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --pretrained_vae_model_name_or_path=$VAE_PATH \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="fp16" \
    --instance_prompt="a photo of $subject" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=500 \
    --rank 4 \
    --validation_prompt="A photo of $subject wearing a red hat" \
    --validation_epochs=40 \
    --seed="0" #  --report_to="wandb" 

done
