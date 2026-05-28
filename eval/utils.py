import os
import sys
import random
import torch
from datasets import load_dataset, Features, Value, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Dict, List, Optional, Any

MODEL_PATHS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "mistral_nemo": "mistralai/Mistral-Nemo-Instruct-2407",
}

DEFAULT_BBH_TASKS = [
    "formal_fallacies",
    "logical_deduction_five_objects",
    "tracking_shuffled_objects_five_objects",
    "causal_judgement",
    "web_of_lies",
]

BBH_INSTRUCTIONS = {
    "formal_fallacies": "Answer with 'valid' or 'invalid'.",
    "logical_deduction_five_objects": "Answer with the option letter in parentheses, e.g., (A).",
    "logical_deduction_seven_objects": "Answer with the option letter in parentheses, e.g., (A).",
    "logical_deduction_three_objects": "Answer with the option letter in parentheses, e.g., (A).",
    "tracking_shuffled_objects_five_objects": "Answer with the option letter in parentheses, e.g., (A).",
    "tracking_shuffled_objects_seven_objects": "Answer with the option letter in parentheses, e.g., (A).",
    "tracking_shuffled_objects_three_objects": "Answer with the option letter in parentheses, e.g., (A).",
    "causal_judgement": "Answer with 'Yes' or 'No'.",
    "web_of_lies": "Answer with 'Yes' or 'No'.",
}


def get_eos_token_ids(tokenizer: AutoTokenizer, model_name: str) -> List[int]:
    """Return all stop-token IDs: model EOS(es) + model-specific extras + <|endofdebate|>."""
    ids: set = set()

    raw = tokenizer.eos_token_id
    if raw is not None:
        ids.update(raw if isinstance(raw, list) else [raw])

    # Llama 3.1 needs both <|end_of_text|> (128001) and <|eot_id|> (128009)
    if model_name == "llama":
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ids.add(eot if (eot is not None and eot != tokenizer.unk_token_id) else 128009)

    # Qwen 2.5 needs both <|endoftext|> (151643) and <|im_end|> (151645)
    elif model_name == "qwen2.5":
        for tok in ("<|im_end|>", "<|endoftext|>"):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                ids.add(tid)

    # Fine-tuned models trained with sft.py have <|endofdebate|> as a debate boundary token
    vocab = tokenizer.get_vocab()
    if "<|endofdebate|>" in vocab:
        ids.add(vocab["<|endofdebate|>"])

    result = sorted(ids)
    print(f"Stop token IDs: {result}")
    return result

def load_model(model_name: str, model_path: Optional[str] = None, device: int = 0):
    """Load model and tokenizer, pinned to a single GPU.

    Args:
        model_name: Key in MODEL_PATHS (llama, qwen2.5, mistral_nemo).
        model_path: Optional path to a PEFT/LoRA adapter directory.
        device: CUDA device index (default 0).

    If model_path is given, loads as a PEFT/LoRA fine-tuned model on top of the
    base identified by model_name. Otherwise loads the base model directly.
    """
    device_map = {"": device}
    cuda_dev = f"cuda:{device}"

    if model_path:
        print(f"Loading fine-tuned model from {model_path} (base: {model_name}, device: {cuda_dev})")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS[model_name],
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        if base.get_input_embeddings().weight.shape[0] != len(tokenizer):
            base.resize_token_embeddings(len(tokenizer))

        from peft import PeftModel
        model = PeftModel.from_pretrained(base, model_path).eval()
    else:
        path = MODEL_PATHS[model_name]
        print(f"Loading base model {model_name} from {path} (device: {cuda_dev})")
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
        tokenizer.padding_side = "left"

        try:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            ).eval()
            print("Loaded with FlashAttention-2")
        except Exception as e:
            print(f"FlashAttention-2 unavailable ({e}), falling back to standard attention")
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
            ).eval()

    eos_ids = get_eos_token_ids(tokenizer, model_name)
    return model, tokenizer, eos_ids, cuda_dev


def generate_responses(
    model, tokenizer, eos_ids: List[int], prompts: List[str],
    max_new_tokens: int, batch_size: int, device: str = "cuda:0",
) -> tuple:
    responses, in_toks, out_toks = [], [], []
    pad_id = tokenizer.pad_token_id

    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        inputs.pop("token_type_ids", None)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_ids,
                pad_token_id=pad_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        for i in range(len(batch)):
            n_in = (inputs["attention_mask"][i] == 1).sum().item()
            text = tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True).strip()
            n_out = len(outputs[i]) - prompt_len
            responses.append(text)
            in_toks.append(n_in)
            out_toks.append(n_out)

    return responses, in_toks, out_toks


def generate_arithmetic_problems(seed: int, num_samples: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    problems = []
    for _ in range(num_samples):
        ns = [rng.randint(10, 99) for _ in range(6)]
        expr = f"{ns[0]}+{ns[1]}*{ns[2]}+{ns[3]}-{ns[4]}*{ns[5]}"
        answer = ns[0] + ns[1] * ns[2] + ns[3] - ns[4] * ns[5]
        problems.append({"question": expr, "answer": str(answer)})
    return problems


def load_benchmark_data(
    benchmark: str, num_samples: int, seed: int,
    bbh_tasks: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    random.seed(seed)

    if benchmark == "gsm":
        ds = load_dataset("gsm8k", "main")["test"]
        return random.sample(list(ds), min(num_samples, len(ds)))

    elif benchmark == "math":
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra")["test"]
        filtered = [x for x in ds if x["level"] in ("Level 1", "Level 2", "Level 3")]
        print(f"MATH: {len(filtered)} items at levels 1-3 (out of {len(ds)})")
        return random.sample(filtered, min(num_samples, len(filtered)))

    elif benchmark == "mmlu":
        feat = Features({
            "question_id": Value("int32"), "question": Value("string"),
            "options": Sequence(Value("string")), "answer": Value("string"),
            "answer_index": Value("int32"), "cot_content": Value("string"),
            "category": Value("string"), "src": Value("string"),
        })
        ds = load_dataset("TIGER-Lab/MMLU-Pro", features=feat)["test"]
        return random.sample(list(ds), min(num_samples, len(ds)))

    elif benchmark == "arithmetic":
        return generate_arithmetic_problems(seed, num_samples)

    elif benchmark == "bbh":
        tasks = bbh_tasks or DEFAULT_BBH_TASKS
        all_qs: List[Dict[str, Any]] = []
        per_task, rem = divmod(num_samples, len(tasks))
        for i, task in enumerate(tasks):
            n = per_task + (1 if i < rem else 0)
            ds = load_dataset("maveriq/bigbenchhard", task)["train"]
            qs = random.sample(list(ds), min(n, len(ds)))
            for q in qs:
                q["task"] = task
            all_qs.extend(qs)
            print(f"BBH {task}: {len(qs)} questions")
        return all_qs

    raise ValueError(f"Unknown benchmark: {benchmark}")


def format_prompt(data: Dict[str, Any], benchmark: str) -> str:
    if benchmark == "gsm":
        q = data["question"]
        return (f"Problem: {q} Explain your reasoning. Your final answer should be a single "
                f"numerical number, in the form \\boxed{{answer}}, at the end of your response. "
                f"For example, \\boxed{{12}}.\n\n")
    elif benchmark == "math":
        q = data["problem"]
        return (f"Problem: {q} Explain your reasoning. Your final answer should be a single "
                f"numerical number, in the form \\boxed{{answer}}, at the end of your response. "
                f"For example, \\boxed{{12}}.\n\n")
    elif benchmark == "mmlu":
        q, opts = data["question"], data["options"]
        opts_str = "".join(f"{chr(65+i)}) {o}\n" for i, o in enumerate(opts))
        return (f"Problem: {q}\n\nOptions:\n{opts_str}\nPlease provide your reasoning and then "
                f"give your final answer in the form \\boxed{{answer}}, where answer is a single "
                f"letter (A, B, C, D, etc.). For example, \\boxed{{A}}.")
    elif benchmark == "arithmetic":
        q = data["question"]
        return (f"Problem: {q} Explain your reasoning. Your final answer should be a single "
                f"numerical number, in the form \\boxed{{answer}}, at the end of your response. "
                f"For example, \\boxed{{12}}.\n\n")
    elif benchmark == "bbh":
        q, task = data["input"], data.get("task", "unknown")
        instruction = BBH_INSTRUCTIONS.get(task, "Provide your answer.")
        return (f"Problem: {q}\n\n{instruction} Explain your reasoning. Your final answer should "
                f"be in the form \\boxed{{answer}}, at the end of your response.\n\n")
    raise ValueError(f"Unknown benchmark: {benchmark}")


def format_revision_prompt(
    question: str, options, other_answers: List[str], benchmark: str
) -> str:
    """Prompt asking an agent to revise its answer given the other agents' responses."""
    prefix = "These are the solutions to the problem from other agents:\n\n"
    for i, ans in enumerate(other_answers):
        prefix += f"Agent {i+1} solution: ```{ans}```\n\n"
    prefix += "Using the solutions from other agents as additional information, can you provide your answer to the problem?\n\n"

    if benchmark == "mmlu":
        opts_str = "".join(f"{chr(65+i)}) {o}\n" for i, o in enumerate(options))
        return (prefix + f"The original problem is: {question}\n\nOptions:\n{opts_str}\n"
                f"Your final answer should be a single letter in the form \\boxed{{answer}}. "
                f"For example, \\boxed{{A}}.")
    elif benchmark == "bbh":
        task = options or "unknown"
        instruction = BBH_INSTRUCTIONS.get(task, "Provide your answer.")
        return (prefix + f"The original problem is: {question}\n\n{instruction} "
                f"Your final answer should be in the form \\boxed{{answer}}.")
    else:
        return (prefix + f"The original problem is: {question}\n\n"
                f"Your final answer should be a single numerical number, in the form "
                f"\\boxed{{answer}}, at the end of your response. For example, \\boxed{{12}}.")


def extract_boxed_answer(text: str) -> Optional[str]:
    last = text.rfind("\\boxed{")
    if last == -1:
        return None
    start = last + 7
    end = text.find("}", start)
    return text[start:end].strip() if end > start else None


def normalize_bbh_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip().lower()
    if answer.startswith("(") and answer.endswith(")"):
        answer = answer[1:-1]
    return answer


def score_answer(pred: Optional[str], gold: str, benchmark: str) -> bool:
    if pred is None:
        return False
    if benchmark == "bbh":
        return normalize_bbh_answer(pred) == normalize_bbh_answer(gold)
    return pred == gold


def extract_gold_answer(data: Dict[str, Any], benchmark: str) -> str:
    if benchmark == "gsm":
        return data["answer"].split("#### ")[1].strip()
    elif benchmark == "math":
        return extract_boxed_answer(data["solution"])
    elif benchmark == "mmlu":
        return data["answer"]
    elif benchmark == "arithmetic":
        return data["answer"]
    elif benchmark == "bbh":
        return data["target"].strip()
    raise ValueError(f"Unknown benchmark: {benchmark}")


def question_text(data: Dict[str, Any], benchmark: str) -> str:
    return data.get("question") or data.get("input") or data.get("problem") or ""


def print_breakdown(results: List[Dict[str, Any]], benchmark: str) -> None:
    correct = sum(r["is_correct"] for r in results)
    print(f"Accuracy: {correct/len(results):.2%} ({correct}/{len(results)})")

    if benchmark == "mmlu":
        by_cat: Dict[str, Dict] = {}
        for r in results:
            c = r["category"]
            by_cat.setdefault(c, {"correct": 0, "total": 0})
            by_cat[c]["total"] += 1
            if r["is_correct"]:
                by_cat[c]["correct"] += 1
        for cat, s in sorted(by_cat.items()):
            print(f"  {cat}: {s['correct']/s['total']:.2%} ({s['correct']}/{s['total']})")

    if benchmark == "bbh":
        by_task: Dict[str, Dict] = {}
        for r in results:
            t = r["task"]
            by_task.setdefault(t, {"correct": 0, "total": 0})
            by_task[t]["total"] += 1
            if r["is_correct"]:
                by_task[t]["correct"] += 1
        for task, s in sorted(by_task.items()):
            print(f"  {task}: {s['correct']/s['total']:.2%} ({s['correct']}/{s['total']})")
