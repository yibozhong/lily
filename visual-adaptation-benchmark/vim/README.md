# Visual Adaptation Benchmark on Vision Mamba

## Environment setup

```bash
conda create -n lily-vtab-vim python=3.10.13
conda activate lily-vtab-vim
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
export CUDA_HOME=/usr/local/cuda
sudo apt install build-essential
pip install causal_conv1d==1.1.0
pip install -e mamba-1p1p1
```

## Datasets

Kindly refer to [RepAdapter](https://github.com/luogen1996/RepAdapter#repadapter) to download the datasets from google drive. Then rename the folder to `vtab-1k` and move it to `lily/visual-adaptation-benchmark`. 

## Pretrained Vim

Download the model from [vim-s](https://huggingface.co/hustvl/Vim-small-midclstok). For the experiment, we use `vim_s_midclstok_80p5acc.pth`. Move the file to `lily/visual-adaptation-benchmark/vim/`

## Run Lily
Run the following script to finetune vim-s on VTAB-1K:

```bash
cd <YOUR PATH>/lily/visual-adaptation-benchmark/vim/
bash lily.sh
```

## Acknowledgments

The code is based on [vision-mamba](https://github.com/hustvl/Vim).



