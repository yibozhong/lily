# Commonsense Reasoning

## Environment Setup

```bash
conda create -n lily-commonsense python=3.8
conda activate lily-commonsense
pip install -r requirements.txt
```

## Datasets

Please refer to [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters). Download folder `dataset` and `ft-training_set` and move them to `lily/commonsense-reasoning/`.

## Fine-tuning

Directly finetuning LLaMA3-8B or Falcon-Mamba-7B by:

```bash
bash finetune_llama.sh
bash finetune_mamba.sh
```

The scripts are only tested in single-GPU setting since the experiments are conducted only on a RTX 4090 GPU.

## Evaluation

After fine-tuning, use following scripts to evaluate the fine-tuned model on commonsense reasoning tasks.

```bash
bash evaluate_llama.sh
bash evaluate_mamba.sh
```

## Acknowledgement

The code is heavily based on [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters).