"""
GRPO Training with Iterative Length Pruning (ThinkPrune approach)

Trains a debate-capable model using GRPO with iterative token-budget pruning.
Starts from an SFT LoRA checkpoint and applies a fresh LoRA for GRPO.

Usage:
    accelerate launch grpo.py \\
        --model_name llama \\
        --sft_model_path ./llama-sft-lora-arith-final \\
        --dataset data/arithmetic_3_2_diverse.json \\
        --output_dir llama-grpo-arith
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import gc
import argparse
import numpy as np
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from typing import Optional


MODEL_CONFIGS = {
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "project_name": "llama-grpo",
        "learning_rate": 5e-6,
        "per_device_batch_size": 2,
        "num_generations": 4,
        "gradient_accumulation_steps": 2,
    },
    "qwen2.5": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "project_name": "qwen2.5-grpo",
        "learning_rate": 5e-6,
        "per_device_batch_size": 2,
        "num_generations": 4,
        "gradient_accumulation_steps": 2,
    },
    "mistral_nemo": {
        "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "project_name": "mistral-nemo-grpo",
        "learning_rate": 3e-6,
        "per_device_batch_size": 1,
        "num_generations": 4,
        "gradient_accumulation_steps": 4,
    },
}

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# Prompt and answer utilities
# ---------------------------------------------------------------------------

def format_prompt(question: str) -> str:
    return f"Problem: {question}\n\n"


def extract_boxed_answer(text: str) -> Optional[str]:
    last = text.rfind('\\boxed{')
    if last == -1:
        return None
    start = last + 7
    end = text.find('}', start)
    return text[start:end].strip() if end > start else None


def clip_completion(completion: str, max_tokens: int, tokenizer) -> str:
    tokens = tokenizer.encode(completion, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return completion
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

DEBATE_WEIGHTS = {
    "<|Round 1 - Initial Solutions|>": 0.3,
    "<|Round 2 - Revision|>": 0.3,
    "<|Agent 1 Solution|>": 0.3,
    "<|Agent 2 Solution|>": 0.3,
    "<|Agent 3 Solution|>": 0.3,
    "<|Agent 1 Revision|>": 0.3,
    "<|Agent 2 Revision|>": 0.3,
    "<|Agent 3 Revision|>": 0.3,
    "<|Consensus|>": 0.5,
    "<|endofdebate|>": 0.3,
}
# These two always receive full reward regardless of pruning stage
_ALWAYS_FULL = {"<|Consensus|>", "<|endofdebate|>"}


def _structure_reward_coeff(max_tokens: int) -> float:
    initial, final = 2048, 500
    if max_tokens >= initial:
        return 1.0
    if max_tokens <= final:
        return 0.05
    progress = (max_tokens - final) / (initial - final)
    return 0.1 + 0.9 * progress


def reward_function(completions, **kwargs):
    tokenizer = kwargs.get('tokenizer')
    max_tokens = kwargs.get('max_tokens', 3000)
    gold_answers = kwargs.get('gold_answer', [None] * len(completions))

    if tokenizer is None:
        raise ValueError("tokenizer must be provided in kwargs")

    coeff = _structure_reward_coeff(max_tokens)
    rewards = []
    orig_lens, clip_lens, correctness_r, structure_r = [], [], [], []
    n_clipped = n_correct = 0

    for i, completion in enumerate(completions):
        orig_len = len(tokenizer.encode(completion, add_special_tokens=False))
        orig_lens.append(orig_len)

        if orig_len > max_tokens:
            completion = clip_completion(completion, max_tokens, tokenizer)
            n_clipped += 1
        clip_lens.append(len(tokenizer.encode(completion, add_special_tokens=False)))

        # Correctness
        gold = gold_answers[i] if isinstance(gold_answers, list) else gold_answers
        pred = extract_boxed_answer(completion)
        if pred == str(gold):
            cr = 1.0
            n_correct += 1
        else:
            cr = 0.0
        correctness_r.append(cr)

        # Structure
        sr = sum(
            w if tok in _ALWAYS_FULL else w * coeff
            for tok, w in DEBATE_WEIGHTS.items()
            if tok in completion
        )
        structure_r.append(sr)
        rewards.append(cr + sr)

    try:
        if wandb.run is not None:
            n = len(completions)
            avg_orig = np.mean(orig_lens)
            avg_clip = np.mean(clip_lens)
            wandb.log({
                "reward/correctness_mean": np.mean(correctness_r),
                "reward/structure_mean": np.mean(structure_r),
                "reward/accuracy": n_correct / n,
                "reward/total_mean": np.mean(rewards),
                "reward/total_std": np.std(rewards),
                "reward/structure_coeff": coeff,
                "length/original_avg": avg_orig,
                "length/clipped_avg": avg_clip,
                "length/clipping_rate": n_clipped / n,
                "length/max_tokens": max_tokens,
                "length/reduction_ratio": (avg_orig - avg_clip) / avg_orig if avg_orig > 0 else 0,
            })
    except Exception:
        pass

    return rewards


def reward_function_wrapper(completions, **kwargs):
    kwargs['tokenizer'] = reward_function_wrapper.tokenizer
    kwargs['max_tokens'] = reward_function_wrapper.max_tokens
    return reward_function(completions, **kwargs)


reward_function_wrapper.tokenizer = None
reward_function_wrapper.max_tokens = 4000


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class WandbCallback(TrainerCallback):
    def __init__(self, log_every_n_steps=10):
        self.log_every_n_steps = log_every_n_steps
        self._step = 0

    def on_step_end(self, args, state, control, **kwargs):
        self._step += 1
        if self._step % self.log_every_n_steps == 0:
            try:
                wandb.log({"train/step": self._step, "train/global_step": state.global_step,
                           "train/epoch": state.epoch})
                if state.log_history and 'loss' in state.log_history[-1]:
                    wandb.log({"train/loss": state.log_history[-1]['loss']})
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(dataset, tokenizer, max_tokens: int):
    def fmt(example):
        prompt = format_prompt(example['question'])
        toks = tokenizer(prompt, truncation=True, max_length=2048)
        return {
            'prompt': prompt,
            'input_ids': toks['input_ids'],
            'attention_mask': toks['attention_mask'],
            'gold_answer': example['gold_answer'],
        }
    return dataset.map(fmt, remove_columns=dataset.column_names, desc="Preparing dataset")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_with_iterative_pruning(
    model, tokenizer, dataset, save_dir_base: str,
    model_config: dict,
    initial_max_tokens: int = 2048,
    final_max_tokens: int = 512,
    num_iterations: int = 3,
    base_model_id: str = None,
):
    token_limits = np.linspace(initial_max_tokens, final_max_tokens, num_iterations, dtype=int)
    current_model = model
    project_name = model_config["project_name"]
    lr = model_config["learning_rate"]
    bsz = model_config["per_device_batch_size"]
    n_gen = model_config["num_generations"]
    grad_accum = model_config["gradient_accumulation_steps"]

    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
    )

    for iteration, max_tokens in enumerate(token_limits):
        print(f"\n=== Iteration {iteration+1}/{num_iterations}  max_tokens={max_tokens} ===")

        reward_function_wrapper.tokenizer = tokenizer
        reward_function_wrapper.max_tokens = int(max_tokens)

        save_dir = f"{save_dir_base}_iter{iteration+1}_tokens{max_tokens}"

        wandb.init(
            project=project_name,
            name=save_dir,
            config={
                "model": base_model_id,
                "iteration": iteration + 1,
                "max_tokens": int(max_tokens),
                "num_iterations": num_iterations,
                "learning_rate": lr,
                "num_generations": n_gen,
                "gradient_accumulation_steps": grad_accum,
                "beta": 0.02,
            },
        )

        train_dataset = prepare_dataset(dataset['train'], tokenizer, int(max_tokens))

        grpo_config = GRPOConfig(
            output_dir=save_dir,
            learning_rate=lr,
            per_device_train_batch_size=bsz,
            num_generations=n_gen,
            gradient_accumulation_steps=grad_accum,
            max_grad_norm=0.3,
            beta=0.02,
            num_iterations=3,
            epsilon=0.1,
            scale_rewards=True,
            loss_type="bnpo",
            mask_truncated_completions=False,
            max_completion_length=2048,
            temperature=0.7,
            top_p=0.9,
            logging_steps=5,
            log_completions=True,
            save_strategy="epoch",
            save_total_limit=3,
            num_train_epochs=2,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            report_to=None,
        )

        trainer = GRPOTrainer(
            model=current_model,
            args=grpo_config,
            train_dataset=train_dataset,
            reward_funcs=reward_function_wrapper,
            peft_config=lora_config if iteration == 0 else None,
        )
        trainer.add_callback(WandbCallback(log_every_n_steps=5))
        trainer.train()

        os.makedirs(save_dir, exist_ok=True)
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        wandb.save(f"{save_dir}/*")
        print(f"Saved to {save_dir}")
        wandb.finish()

        del trainer
        torch.cuda.empty_cache()
        gc.collect()

        if iteration < num_iterations - 1:
            print("Merging adapter and reloading for next iteration...")
            base = AutoModelForCausalLM.from_pretrained(
                base_model_id, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True,
            )
            if base.get_input_embeddings().weight.shape[0] != len(tokenizer):
                base.resize_token_embeddings(len(tokenizer))
            current_model = PeftModel.from_pretrained(base, save_dir).merge_and_unload()
            current_model = torch.compile(current_model) if False else current_model

    return current_model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO training with iterative length pruning")
    parser.add_argument("--model_name", choices=list(MODEL_CONFIGS), default="llama")
    parser.add_argument("--sft_model_path", type=str, required=True,
                        help="Path to SFT LoRA adapter directory (output of sft.py)")
    parser.add_argument("--dataset", type=str, default="data/arithmetic_3_2_diverse.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base output directory stem (default: <model_name>-grpo-arith)")
    parser.add_argument("--initial_max_tokens", type=int, default=2048)
    parser.add_argument("--final_max_tokens", type=int, default=512)
    parser.add_argument("--num_iterations", type=int, default=3)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_name]
    base_model_id = cfg["model_id"]
    output_dir = args.output_dir or f"{args.model_name}-grpo-arith"

    print(f"Model: {args.model_name}  ({base_model_id})")
    print(f"SFT model: {args.sft_model_path}")
    print(f"Dataset: {args.dataset}")

    dataset = load_dataset('json', data_files=args.dataset)

    # Load tokenizer from SFT model (includes custom debate tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Load base model, resize to match SFT vocab, merge SFT adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, args.sft_model_path)
    model = model.merge_and_unload()
    print("SFT adapter merged into base model")

    train_with_iterative_pruning(
        model, tokenizer, dataset, output_dir,
        model_config=cfg,
        initial_max_tokens=args.initial_max_tokens,
        final_max_tokens=args.final_max_tokens,
        num_iterations=args.num_iterations,
        base_model_id=base_model_id,
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
