# Subject Driven Generation
## Environment Setup

```bash
conda create -n lily-generation python=3.12
conda activate lily-generation
pip install -r requirements.txt
```

To install the diffusers:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

## Datasets

Download the datasets from [dreambooth](https://github.com/google/dreambooth/tree/main/dataset). Move the folder to `lily/subject-driven-generation/`.

## Training

Use the scripts to fine-tune the stable-diffusion-xl-base-1.0 model with Lily or LoRA:

```bash
bash train_lily.sh
bash train_lora.sh
```

## Generation

After fine-tuning, use the following scripts to generate images for an given subject:

```bash
bash infer_lily.sh
bash infer_lora.sh
```

The subject can be changed in `infer_lily.py` or `infer_lora.py`

## Acknowledgments

The code is based on [moslora](https://github.com/wutaiqiang/MoSLoRA/tree/main/subject_driven_generation) and [diffusers](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md).

