
# Visual Adaptation Benchmark on Vision Transformer

## Environment Setup

```bash
conda create -n lily-vatb python=3.8
conda activate lily-vtab
pip install -r requirements.txt
```

## Datasets

Kindly refer to [RepAdapter](https://github.com/luogen1996/RepAdapter#repadapter) to download the datasets from google drive. Then rename the folder to `vtab-1k` and move it to `lily/visual-adaptation-benchmark`.

## Pretrained ViT

Get the model weight from [ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) to `lily/visual-adaptation-benchmark/vit`.

## Run Lily

```bash
cd <YOUR PATH>/lily/visual-adaptation-benchmark/vit/
bash lily.sh
```

## Acknowledgments

The code is based on [FacT](https://github.com/JieShibo/PETL-ViT/tree/main/FacT), [NOAH](https://github.com/ZhangYuanhan-AI/NOAH) and [timm](https://github.com/rwightman/pytorch-image-models).
