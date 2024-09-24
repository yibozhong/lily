# Natural Language Understanding

## Environment setup

```bash
conda create -n lily-nlu python=3.12
conda activate lily-nlu
pip install -r requirements.txt
```

## Fine-tune

Use the script to finetune with lily:

```bash
bash run.sh
```

The script already has the best configuration from our experiments. However, the hyperparameters can be flexibly modified.

## Acknowledgments

The code is based on [fourierft](https://github.com/Chaos96/fourierft/tree/f8ab847bd7e7cb2f6a469bc5c8577fe96e5362bd/experiments/GLUE).