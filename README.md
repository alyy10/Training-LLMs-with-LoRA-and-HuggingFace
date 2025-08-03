# Fine-Tuning FLAN-T5-XXL with LoRA and Hugging Face for Summarization

This project demonstrates how to fine-tune the FLAN-T5-XXL model using Low-Rank Adaptation (LoRA) and 8-bit quantization with Hugging Face libraries to perform abstractive summarization on the [samsum dataset](https://huggingface.co/datasets/samsum). By leveraging parameter-efficient fine-tuning, this approach significantly reduces memory requirements while achieving strong performance, with a ROUGE-1 score of 50.38% on the test dataset, surpassing full fine-tuning of a smaller FLAN-T5-base model.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project fine-tunes the FLAN-T5-XXL model (11B parameters) using LoRA and 8-bit quantization to efficiently train a large language model for text summarization. The samsum dataset, containing ~16k messenger-like conversations with summaries, is used for training and evaluation. The fine-tuned model achieves a ROUGE-1 score of 50.38% with a compact LoRA checkpoint of only 84MB, compared to a full fine-tuning setup requiring 8x A100 40GB GPUs and costing ~$322.

## Dataset

The [samsum dataset](https://huggingface.co/datasets/samsum) consists of ~14,732 training samples and 819 test samples, each containing a dialogue and a corresponding summary. Example:

```json
{
  "id": "13818513",
  "summary": "Amanda baked cookies and will bring Jerry some tomorrow.",
  "dialogue": "Amanda: I baked cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)"
}
```

## Requirements

- Python 3.10+
- CUDA-enabled GPU with at least 18GB memory for inference
- PyTorch with CUDA support
- Hugging Face libraries (`transformers`, `datasets`, `peft`, `evaluate`, `bitsandbytes`)
- Additional dependencies: `rouge-score`, `tensorboard`, `py7zr`

## Installation

Clone the repository and install the required dependencies:

The `requirements.txt` file should include:

```
peft==0.2.0
transformers==4.27.2
datasets==2.9.0
accelerate==0.17.1
evaluate==0.4.0
bitsandbytes==0.37.1
loralib
rouge-score
tensorboard
py7zr
```

Alternatively, run the following commands as shown in the notebook:

```bash
pip install "peft==0.2.0"
pip install "transformers==4.27.2" "datasets==2.9.0" "accelerate==0.17.1" "evaluate==0.4.0" "bitsandbytes==0.37.1" loralib --upgrade --quiet
pip install rouge-score tensorboard py7zr
```

## Usage

Follow these steps to run the project:

1. **Setup Environment**: Install dependencies as described above. The notebook assumes a PyTorch Deep Learning AMI with CUDA drivers and PyTorch pre-installed.
2. **Load and Preprocess Dataset**:

   - Load the samsum dataset using `datasets.load_dataset("samsum")`.
   - Tokenize dialogues and summaries using the FLAN-T5-XXL tokenizer.
   - Preprocess the dataset to create input-output pairs with a maximum source length of 255 tokens and a maximum target length of 50 tokens (based on 85th and 90th percentiles, respectively).
   - Save the tokenized dataset to disk for efficient loading.
3. **Fine-Tune Model**:

   - Load the sharded FLAN-T5-XXL model (`philschmid/flan-t5-xxl-sharded-fp16`) with 8-bit quantization to reduce memory usage.
   - Apply LoRA with the following configuration:
     ```python
     lora_config = LoraConfig(
         r=16,
         lora_alpha=32,
         target_modules=["q", "v"],
         lora_dropout=0.05,
         bias="none",
         task_type=TaskType.SEQ_2_SEQ_LM
     )
     ```
   - Train the model for 5 epochs with a learning rate of 1e-3 using `Seq2SeqTrainer`.
   - Save the LoRA model and tokenizer to the `results` directory.
4. **Inference and Evaluation**:

   - Load the fine-tuned LoRA model and tokenizer.
   - Generate summaries for test dataset samples.
   - Evaluate using ROUGE metrics (`rouge1`, `rouge2`, `rougeL`, `rougeLsum`).


For a single inference example:

```python
from datasets import load_dataset
from random import randrange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Load dataset and model
dataset = load_dataset("samsum")
peft_model_id = "results"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})

# Get a random sample
sample = dataset['test'][randrange(len(dataset["test"]))]
input_ids = tokenizer(sample["dialogue"], return_tensors="pt", truncation=True).input_ids.cuda()
outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
print(f"Input: {sample['dialogue']}\n{'---'*20}")
print(f"Summary: {tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")
```

## Results

The fine-tuned model achieves the following ROUGE scores on the samsum test dataset:

- **ROUGE-1**: 50.39%
- **ROUGE-2**: 24.84%
- **ROUGE-L**: 41.37%
- **ROUGE-Lsum**: 41.39%

These results represent a ~3% improvement in ROUGE-1 over a fully fine-tuned FLAN-T5-base model (47.23%), with the LoRA checkpoint being only 84MB compared to the full model’s size. The fine-tuning process trains only 0.17% of the model’s parameters (18.87M out of 11.15B), making it highly memory-efficient.

