import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

MODEL = "HuggingFaceTB/SmolLM2-135M"
dataset = load_dataset("EleutherAI/SmolLM2-135M-10B", split="train", streaming=True)
dataset = dataset.take(1000)  # Only processes first 1000

# Convert IterableDataset to Dataset for chunk_and_tokenize compatibility
from datasets import Dataset
dataset = Dataset.from_dict({"text": [item["text"] for item in dataset]})

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenized = chunk_and_tokenize(dataset, tokenizer)


gpt = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": "cuda:1"},
    torch_dtype=torch.bfloat16,
)

cfg = TrainConfig(
    SaeConfig(transcode=False),
    batch_size=16,
    save_dir="checkpoints/test_sae",
    run_name="minimal_test",
    log_to_wandb=True,
    wandb_log_frequency=1
)
trainer = Trainer(cfg, tokenized, gpt)

trainer.fit()