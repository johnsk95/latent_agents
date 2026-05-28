"""
rouge_analysis.py

ROUGE fidelity analysis for internalized diverse agents.

Measures whether a fine-tuned IMAD model has internalized agent-specific
personas by comparing:
  - Steered base model  →  ROUGE vs ground-truth agent response
  - Steered IMAD model  →  ROUGE vs ground-truth agent response

A higher ROUGE score for the IMAD model indicates the training caused the model
to genuinely internalize agent personas rather than just following steering at the
activation level.

Usage:
  python steering/rouge_analysis.py \\
      --model_name llama \\
      --imad_model_path ./llama-grpo-arith-final \\
      --vectors_path steering/llama_steering_vectors_diverse.pt \\
      --pairs_path data/agent_steering_pairs_diverse_test.json \\
      --agent_id 3 \\
      --multiplier 5.0 \\
      --layer 20 \\
      --num_samples 100 \\
      --output_dir results/rouge
"""

import os
import gc
import json
import random
import argparse
import torch
from tqdm import tqdm
from typing import Callable
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate


MODEL_PATHS = {
    'llama':        'meta-llama/Llama-3.1-8B-Instruct',
    'qwen2.5':      'Qwen/Qwen2.5-7B-Instruct',
    'mistral_nemo': 'mistralai/Mistral-Nemo-Instruct-2407',
}

GEN_CONFIG = {
    'max_new_tokens': 512,
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.9,
    'use_cache': False,   # must be False for steering hooks to work correctly
}


# ---------------------------------------------------------------------------
# Activation addition hook (CAA)
# ---------------------------------------------------------------------------

class ActivationAdditionHook:
    """Pre-hook that adds a scaled steering vector to the residual stream."""

    def __init__(self, steering_vector: torch.Tensor, multiplier: float,
                 prompt_length_fn: Callable[[], int]):
        self.vector = steering_vector
        self.multiplier = multiplier
        self.prompt_length_fn = prompt_length_fn

    def __call__(self, module, module_input, module_kwargs):
        hidden = module_input[0]
        start  = self.prompt_length_fn()
        vec    = self.vector.to(hidden.device, dtype=hidden.dtype)
        pos_ids = module_kwargs.get('position_ids')
        if pos_ids is not None:
            mask = (pos_ids >= start).unsqueeze(-1).to(hidden.device)
            hidden = hidden + mask.float() * (vec * self.multiplier)
        else:
            hidden[:, start:, :] += vec * self.multiplier
        return (hidden,) + module_input[1:], module_kwargs


def get_layer(model, layer_idx: int):
    try:
        return model.model.model.layers[layer_idx]
    except AttributeError:
        return model.model.layers[layer_idx]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_base_model(model_id: str, device: str):
    print(f"Loading base model {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = 'left'
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    ).eval()
    return mdl, tok


def _load_imad_model(model_name: str, imad_path: str, device: str):
    base_id = MODEL_PATHS[model_name]
    print(f"Loading IMAD model from {imad_path} (base: {base_id})...")
    tok = AutoTokenizer.from_pretrained(imad_path)
    tok.padding_side = 'left'
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )
    if base.get_input_embeddings().weight.shape[0] != len(tok):
        base.resize_token_embeddings(len(tok))
    try:
        mdl = PeftModel.from_pretrained(base, imad_path).eval()
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            imad_path, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True,
        ).eval()
    return mdl, tok


# ---------------------------------------------------------------------------
# Steered generation
# ---------------------------------------------------------------------------

def generate_steered(model, tokenizer, context: str, vector: torch.Tensor,
                     layer_idx: int, multiplier: float, device: str) -> str:
    inputs = tokenizer(context, return_tensors='pt').to(device)
    prompt_len = inputs['input_ids'].shape[1]

    hook = ActivationAdditionHook(vector, multiplier, lambda: prompt_len)
    handle = get_layer(model, layer_idx).register_forward_pre_hook(
        hook, with_kwargs=True
    )
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs, **GEN_CONFIG,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='ROUGE fidelity analysis for internalized diverse agents'
    )
    parser.add_argument('--model_name', choices=list(MODEL_PATHS), default='llama')
    parser.add_argument('--imad_model_path', required=True,
                        help='Path to fine-tuned IMAD model (LoRA adapter or merged)')
    parser.add_argument('--vectors_path', required=True,
                        help='Path to steering vectors .pt (from extract_vectors.py imad mode)')
    parser.add_argument('--pairs_path', required=True,
                        help='Path to test steering pairs JSON')
    parser.add_argument('--agent_id', type=str, default='3',
                        help='Agent to steer for (1, 2, or 3)')
    parser.add_argument('--vector_key', type=str, default=None,
                        help='Steering vector key (default: A{agent_id}_vs_others)')
    parser.add_argument('--multiplier', type=float, default=5.0)
    parser.add_argument('--layer', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='results/rouge')
    args = parser.parse_args()

    vector_key = args.vector_key or f'A{args.agent_id}_vs_others'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Target agent: {args.agent_id}  |  vector: {vector_key}")
    print(f"Multiplier: {args.multiplier}  |  layer: {args.layer}")

    # Load steering vector
    data = torch.load(args.vectors_path, map_location=device)
    vectors = data['vectors']
    if vector_key not in vectors:
        raise ValueError(f"Key '{vector_key}' not in {list(vectors.keys())}")
    sv = vectors[vector_key].to(device)
    print(f"Steering vector norm: {torch.linalg.norm(sv).item():.4f}")

    # Load test pairs
    with open(args.pairs_path) as f:
        all_pairs = json.load(f)
    agent_pairs = [p for p in all_pairs if p['positive_agent_id'] == args.agent_id]
    random.seed(args.seed)
    test_pairs = random.sample(agent_pairs, min(args.num_samples, len(agent_pairs)))
    print(f"Test pairs: {len(test_pairs)} (agent {args.agent_id})")

    references = [p['positive_response'] for p in test_pairs]
    rouge = evaluate.load('rouge')

    # --- Base model ---
    base_model_id = MODEL_PATHS[args.model_name]
    base_mdl, base_tok = _load_base_model(base_model_id, device)
    base_preds = [
        generate_steered(base_mdl, base_tok, p['context'], sv,
                         args.layer, args.multiplier, device)
        for p in tqdm(test_pairs, desc='Base model')
    ]
    del base_mdl, base_tok
    torch.cuda.empty_cache()
    gc.collect()

    # --- IMAD model ---
    imad_mdl, imad_tok = _load_imad_model(args.model_name, args.imad_model_path, device)
    imad_preds = [
        generate_steered(imad_mdl, imad_tok, p['context'], sv,
                         args.layer, args.multiplier, device)
        for p in tqdm(test_pairs, desc='IMAD model')
    ]
    del imad_mdl, imad_tok
    torch.cuda.empty_cache()
    gc.collect()

    # --- Scores ---
    base_scores = rouge.compute(predictions=base_preds, references=references)
    imad_scores = rouge.compute(predictions=imad_preds, references=references)

    print('\n' + '=' * 60)
    print(f'ROUGE COMPARISON — Agent {args.agent_id}  c={args.multiplier}')
    print('=' * 60)
    print(f"{'Metric':<10} {'Base':>10} {'IMAD':>10} {'Delta':>10}")
    print('-' * 60)
    for m in ('rouge1', 'rouge2', 'rougeL'):
        base_v = base_scores[m]
        imad_v = imad_scores[m]
        delta  = (imad_v - base_v) / base_v * 100 if base_v > 0 else float('nan')
        print(f"{m:<10} {base_v:>10.4f} {imad_v:>10.4f} {delta:>+9.2f}%")
    print('=' * 60)

    confirmed = all(imad_scores[m] > base_scores[m] for m in ('rouge1', 'rouge2', 'rougeL'))
    print('HYPOTHESIS CONFIRMED' if confirmed else 'HYPOTHESIS NOT CONFIRMED')

    # Save
    fname = f'rouge_A{args.agent_id}_{vector_key}_mult{args.multiplier}.json'
    out_path = os.path.join(args.output_dir, fname)
    with open(out_path, 'w') as f:
        json.dump({
            'config': {
                'model_name': args.model_name,
                'imad_model_path': args.imad_model_path,
                'agent_id': args.agent_id,
                'vector_key': vector_key,
                'multiplier': args.multiplier,
                'layer': args.layer,
                'num_samples': len(test_pairs),
            },
            'base_model_scores':  {m: base_scores[m] for m in ('rouge1', 'rouge2', 'rougeL')},
            'imad_model_scores':  {m: imad_scores[m] for m in ('rouge1', 'rouge2', 'rougeL')},
            'improvements_pct': {
                m: (imad_scores[m] - base_scores[m]) / base_scores[m] * 100
                for m in ('rouge1', 'rouge2', 'rougeL') if base_scores[m] > 0
            },
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
