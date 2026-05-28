"""
Single-model evaluation for baseline and fine-tuned models (sft, debategpt, imad).

Usage:
  # Base model
  python eval/eval.py --model_name llama --benchmark gsm

  # Fine-tuned model (LoRA adapter)
  python eval/eval.py --model_name llama --model_path ./llama-sft-lora-arith-final --benchmark gsm

Benchmarks: gsm, math, mmlu, arithmetic, bbh
Models:     llama, qwen2.5, mistral_nemo
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import json
import random
import argparse
from typing import Dict, List, Any, Optional

from utils import (
    MODEL_PATHS, DEFAULT_BBH_TASKS,
    load_model, load_benchmark_data, generate_responses,
    format_prompt, extract_gold_answer, extract_boxed_answer,
    score_answer, question_text, print_breakdown,
)


def run_eval(
    model, tokenizer, eos_ids, questions: List[Dict[str, Any]],
    benchmark: str, max_new_tokens: int, batch_size: int, device: str = "cuda:0",
) -> List[Dict[str, Any]]:
    prompts = [format_prompt(d, benchmark) for d in questions]
    responses, in_toks, out_toks = generate_responses(
        model, tokenizer, eos_ids, prompts, max_new_tokens, batch_size, device
    )

    results = []
    for i, data in enumerate(questions):
        gold = extract_gold_answer(data, benchmark)
        pred = extract_boxed_answer(responses[i])
        result = {
            "question": question_text(data, benchmark),
            "gold_answer": gold,
            "model_response": responses[i],
            "model_answer": pred,
            "is_correct": score_answer(pred, gold, benchmark),
            "num_input_tokens": in_toks[i],
            "num_output_tokens": out_toks[i],
        }
        if benchmark == "mmlu":
            result["options"] = data["options"]
            result["category"] = data.get("category", "unknown")
        if benchmark == "bbh":
            result["task"] = data.get("task", "unknown")
        results.append(result)
    return results


def build_save_dict(args, run: int, seed: int, results: List[Dict]) -> Dict:
    correct = sum(r["is_correct"] for r in results)
    save: Dict = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "benchmark": args.benchmark,
        "run": run,
        "seed": seed,
        "accuracy": correct / len(results),
        "correct": correct,
        "total": len(results),
        "results": results,
    }

    if args.benchmark == "mmlu":
        by_cat: Dict = {}
        for r in results:
            c = r["category"]
            by_cat.setdefault(c, {"correct": 0, "total": 0})
            by_cat[c]["total"] += 1
            if r["is_correct"]:
                by_cat[c]["correct"] += 1
        save["category_accuracies"] = by_cat

    if args.benchmark == "bbh":
        by_task: Dict = {}
        for r in results:
            t = r["task"]
            by_task.setdefault(t, {"correct": 0, "total": 0})
            by_task[t]["total"] += 1
            if r["is_correct"]:
                by_task[t]["correct"] += 1
        save["task_accuracies"] = by_task
        save["bbh_tasks"] = args.bbh_tasks or DEFAULT_BBH_TASKS

    return save


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single model on reasoning benchmarks")
    parser.add_argument("--model_name", choices=list(MODEL_PATHS), default="llama",
                        help="Base model architecture")
    parser.add_argument("--model_path", default=None,
                        help="Path to fine-tuned LoRA adapter dir. Omit for base model.")
    parser.add_argument("--benchmark", choices=["gsm", "math", "mmlu", "arithmetic", "bbh"],
                        default="gsm")
    parser.add_argument("--bbh_tasks", nargs="+", default=None,
                        help="BBH tasks to evaluate (default: 5 standard tasks)")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of independent runs with different seeds")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--output_name", default=None,
                        help="Output filename stem (default: <model>-<benchmark>-run<N>.json)")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index (default: 0)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer, eos_ids, cuda_dev = load_model(args.model_name, args.model_path, args.device)

    for run in range(1, args.num_runs + 1):
        seed = args.seed + run - 1
        random.seed(seed)
        print(f"\n=== Run {run}/{args.num_runs} (seed={seed}) ===")

        questions = load_benchmark_data(
            args.benchmark, args.num_samples, seed,
            bbh_tasks=args.bbh_tasks if args.benchmark == "bbh" else None,
        )
        print(f"Loaded {len(questions)} questions from {args.benchmark}")

        results = run_eval(
            model, tokenizer, eos_ids, questions,
            args.benchmark, args.max_new_tokens, args.batch_size, cuda_dev,
        )
        print_breakdown(results, args.benchmark)

        if args.output_name:
            fname = (f"{args.output_name}_run{run}.json" if args.num_runs > 1
                     else f"{args.output_name}.json")
        else:
            label = args.model_path.rstrip("/").split("/")[-1] if args.model_path else args.model_name
            fname = f"{label}-{args.benchmark}-run{run}.json"

        save = build_save_dict(args, run, seed, results)
        out_path = os.path.join(args.output_dir, fname)
        with open(out_path, "w") as f:
            json.dump(save, f, indent=2)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
