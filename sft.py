import os
import torch
import argparse
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AddedToken,
    TrainingArguments,
    Trainer,
)
from typing import Dict, Sequence, List

DEBATE_SPECIAL_TOKENS = [
    '<|Round 1 - Initial Solutions|>',
    '<|Round 2 - Revision|>',
    '<|Agent 1 Solution|>',
    '<|Agent 2 Solution|>',
    '<|Agent 3 Solution|>',
    '<|Agent 1 Revision|>',
    '<|Agent 2 Revision|>',
    '<|Agent 3 Revision|>',
    '<|Consensus|>',
    '<|endofdebate|>',
]

def get_model_config(model_name: str) -> Dict:
    model_configs = {
        "llama": {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "trust_remote_code": True,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "max_length": 2048,
            "learning_rate": 5e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 3,
        },
        "qwen2.5": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "trust_remote_code": True,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "max_length": 2048,
            "learning_rate": 5e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 3,
        },
        "mistral_nemo": {
            "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
            "trust_remote_code": True,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "max_length": 2048,
            "learning_rate": 3e-5,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "num_train_epochs": 6,
        },
    }

    if model_name not in model_configs:
        raise ValueError(f"Model '{model_name}' not supported. Available models: {list(model_configs.keys())}")

    return model_configs[model_name]

def split_debate_into_rounds(debate_trace: str) -> List[str]:
    """Split a debate trace into individual rounds."""
    rounds = []
    current_round = []

    for line in debate_trace.split('\n'):
        if line.startswith('Round '):
            if current_round:
                rounds.append('\n'.join(current_round))
            current_round = [line]
        else:
            current_round.append(line)

    if current_round:
        rounds.append('\n'.join(current_round))

    return rounds

def format_round(question: str, rounds: List[str], round_idx: int) -> str:
    """Format a specific round with context from previous rounds."""
    context = f"Question: {question}\n\n"
    for i in range(round_idx + 1):
        context += rounds[i] + "\n\n"
    return context.strip()

class DebateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length=2048, use_full=False):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_full = use_full

        for example in dataset:
            question = example['question']

            if self.use_full:
                text = f"Problem: {question}\n\n{example['debate_trace']}"
                self.examples.append({
                    'text': text,
                    'question': question,
                    'round_idx': 0,
                    'total_rounds': 1
                })
            else:
                rounds = split_debate_into_rounds(example['debate_trace'])
                for round_idx in range(len(rounds)):
                    text = format_round(question, rounds, round_idx)
                    if len(tokenizer.encode(text)) <= max_length:
                        self.examples.append({
                            'text': text,
                            'question': question,
                            'round_idx': round_idx,
                            'total_rounds': len(rounds)
                        })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        if text.startswith('Problem:'):
            prompt_end = text.find('\n\n')
            if prompt_end != -1:
                prompt_end += 2
            else:
                prompt_end = 0
        else:
            prompt_end = 0
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"][0]
        labels = input_ids.clone()
        if prompt_end > 0:
            prompt_ids = self.tokenizer(
                text[:prompt_end],
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )["input_ids"][0]
            labels[:len(prompt_ids)] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": encodings["attention_mask"][0],
            "labels": labels
        }

def main():
    parser = argparse.ArgumentParser(description="SFT training for debate-style reasoning")
    parser.add_argument("--model", type=str, default="llama",
                       choices=["llama", "qwen2.5", "mistral_nemo"],
                       help="Model to use for training")
    parser.add_argument("--dataset", type=str, default="data/arithmetic_3_2_new.json",
                       help="Path to the training dataset (JSON file)")
    parser.add_argument("--output_suffix", type=str, default="arith",
                       help="Suffix for output directory (e.g., 'arith' -> 'llama-sft-lora-arith')")

    args = parser.parse_args()

    model_name = args.model
    model_config = get_model_config(model_name)
    print(f"Using model: {model_name} ({model_config['model_id']})")

    dataset = load_dataset('json', data_files=args.dataset)
    print(f"Loaded dataset from: {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_id"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    else:
        print(f"Pad token already set: {tokenizer.pad_token}")

    tokenizer.padding_side = "right"

    special_tokens = [
        AddedToken(tag, single_word=False, lstrip=False, rstrip=False, normalized=False)
        for tag in DEBATE_SPECIAL_TOKENS
    ]
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    print(f"Added {num_added} debate special tokens. Vocab size: {len(tokenizer)}")

    train_dataset = DebateDataset(dataset['train'], tokenizer, max_length=model_config["max_length"], use_full=True)
    print(f"Created {len(train_dataset)} training examples from {len(dataset['train'])} debates")

    print(f"Loading model {model_config['model_id']}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_id"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=model_config["trust_remote_code"],
    )

    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_vocab_size = len(tokenizer)
    print(f"Resized model embeddings to vocab size: {new_vocab_size}")

    # Initialize new token embeddings with mean of existing embeddings to avoid
    # random-init noise being amplified by models that scale embeddings.
    if new_vocab_size > original_vocab_size:
        with torch.no_grad():
            embed_weight = model.get_input_embeddings().weight
            mean_embed = embed_weight[:original_vocab_size].mean(dim=0)
            for i in range(original_vocab_size, new_vocab_size):
                embed_weight[i] = mean_embed
            lm_head = model.get_output_embeddings()
            if lm_head is not None and lm_head.weight.data_ptr() != embed_weight.data_ptr():
                lm_mean = lm_head.weight[:original_vocab_size].mean(dim=0)
                for i in range(original_vocab_size, new_vocab_size):
                    lm_head.weight[i] = lm_mean
        print(f"Initialized {new_vocab_size - original_vocab_size} new token embeddings with mean of existing embeddings")

    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=model_config["lora_target_modules"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        # Train embed_tokens and lm_head so new special token embeddings receive gradients.
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    output_dir = f"./{model_name}-sft-lora-{args.output_suffix}"
    print(f"Output directory: {output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=model_config["num_train_epochs"],
        per_device_train_batch_size=model_config["batch_size"],
        per_device_eval_batch_size=model_config["batch_size"],
        gradient_accumulation_steps=model_config["gradient_accumulation_steps"],
        learning_rate=model_config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        save_only_model=True,
        remove_unused_columns=False,
        max_grad_norm=model_config.get("max_grad_norm", 0.3),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        report_to=["none"],
        dataloader_drop_last=True,
    )

    pad_id = tokenizer.pad_token_id
    _pad_seq = torch.nn.utils.rnn.pad_sequence
    def collate_fn(batch):
        input_ids = _pad_seq([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
        attention_mask = _pad_seq([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
        labels = _pad_seq([b["labels"] for b in batch], batch_first=True, padding_value=-100)
        seq_len = input_ids.shape[1]
        pad_to = ((seq_len + 7) // 8) * 8
        if pad_to > seq_len:
            p = pad_to - seq_len
            input_ids = torch.nn.functional.pad(input_ids, (0, p), value=pad_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, p), value=0)
            labels = torch.nn.functional.pad(labels, (0, p), value=-100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    trainer.train()

    trainer.save_model(f"{output_dir}-final")
    tokenizer.save_pretrained(f"{output_dir}-final")

if __name__ == "__main__":
    main()
