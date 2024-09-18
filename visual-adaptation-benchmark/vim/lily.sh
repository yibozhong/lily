#!/bin/bash

for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
do
    for DIM in 8
        do
            CUDA_VISIBLE_DEVICES=0 python main.py \
                                --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
                                --batch-size 64 \
                                --drop-path 0.05 \
                                --weight-decay 1e-3 \
                                --lr 1e-2 \
                                --num_workers 1 \
                                --data-path ../vtab-1k/ \
                                --output_dir ./output/vim_s \
                                --no_amp \
                                --seed 42 \
                                --data_set $DATASET \
                                --finetune ./vim_s_midclstok_80p5acc.pth \
                                --backbone vim \
                                --adapt_delta \
                                --adapt_in \
                                --dropout \
                                --ne 5 \
                                --in_dim $DIM 
        done
    done