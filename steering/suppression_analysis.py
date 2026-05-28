"""
suppression_analysis.py

Malicious agent suppression via contrastive activation steering.

Applies a pre-extracted malicious steering vector with a negative multiplier
to suppress evil / hallucination behavior during generation.  Optionally
computes perplexity of generated text using a separate reference model.

The output JSON is intended as input to an LLM judge that rates whether the
model's behavior was successfully suppressed.

Usage:
  # Suppress evil behavior (multiplier < 0 pushes away from evil direction)
  python steering/suppression_analysis.py \\
      --model_name llama \\
      --vectors_path steering/llama_steering_vector_evil.pt \\
      --data_path data/evil_test.json \\
      --task evil \\
      --multiplier -3.0 \\
      --layer 20 \\
      --output_dir results/suppression

  # With perplexity scoring
  python steering/suppression_analysis.py \\
      --model_name mistral_nemo \\
      --vectors_path steering/mistral_nemo_steering_vector_fake.pt \\
      --data_path data/hallu_test.json \\
      --task fake \\
      --multiplier -2.0 \\
      --perplexity \\
      --perplexity_model google/gemma-3-27b-it \\
      --output_dir results/suppression
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from typing import Callable, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


MODEL_PATHS = {
    'llama':        'meta-llama/Llama-3.1-8B-Instruct',
    'qwen2.5':      'Qwen/Qwen2.5-7B-Instruct',
    'mistral_nemo': 'mistralai/Mistral-Nemo-Instruct-2407',
}

GEN_CONFIG = {
    'max_new_tokens': 1024,
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.9,
    'use_cache': False,   # must be False for steering to work correctly
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
        hidden  = module_input[0]
        start   = self.prompt_length_fn()
        vec     = self.vector.to(hidden.device, dtype=hidden.dtype)
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

def load_inference_model(model_name: str, sft_model_path: Optional[str], device: str):
    base_id = MODEL_PATHS[model_name]
    if sft_model_path:
        print(f"Loading fine-tuned model from {sft_model_path} (base: {base_id})...")
        tok = AutoTokenizer.from_pretrained(sft_model_path)
        tok.padding_side = 'left'
        base = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True,
        )
        if base.get_input_embeddings().weight.shape[0] != len(tok):
            base.resize_token_embeddings(len(tok))
        try:
            mdl = PeftModel.from_pretrained(base, sft_model_path).eval()
        except Exception as e:
            print(f"PEFT load failed ({e}), loading as full model")
            mdl = AutoModelForCausalLM.from_pretrained(
                sft_model_path, torch_dtype=torch.bfloat16,
                device_map=device, trust_remote_code=True,
            ).eval()
    else:
        print(f"Loading base model {base_id}...")
        tok = AutoTokenizer.from_pretrained(base_id)
        tok.padding_side = 'left'
        mdl = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True,
        ).eval()
    return mdl, tok


def load_perplexity_model(model_name: str, device: str = 'auto'):
    print(f"Loading perplexity model: {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True,
            attn_implementation='flash_attention_2',
        )
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True,
        )
    mdl.eval()
    return mdl, tok


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(full_text: str, prompt_text: str,
                       ppl_model, ppl_tokenizer) -> float:
    full_ids   = ppl_tokenizer(full_text,   return_tensors='pt').input_ids.to(ppl_model.device)
    prompt_len = ppl_tokenizer(prompt_text, return_tensors='pt').input_ids.shape[1]
    if full_ids.shape[1] <= prompt_len:
        return float('inf')
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100
    try:
        with torch.no_grad():
            loss = ppl_model(input_ids=full_ids, labels=labels, use_cache=False).loss
        return torch.exp(loss).item()
    except Exception:
        return float('inf')


# ---------------------------------------------------------------------------
# Steered generation
# ---------------------------------------------------------------------------

def generate_steered(model, tokenizer, question: str,
                     vector: torch.Tensor, layer_idx: int, multiplier: float,
                     device: str,
                     ppl_model=None, ppl_tokenizer=None) -> dict:
    prompt = f"Problem: {question}\n\n"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
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
        full_resp = tokenizer.decode(out[0], skip_special_tokens=False)
        gen_resp  = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    finally:
        handle.remove()

    ppl = None
    if ppl_model is not None:
        ppl = compute_perplexity(full_resp, prompt, ppl_model, ppl_tokenizer)

    return {'prompt': prompt, 'generated_response': gen_resp, 'perplexity': ppl}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Malicious agent suppression via contrastive activation steering'
    )
    parser.add_argument('--model_name', choices=list(MODEL_PATHS), default='llama')
    parser.add_argument('--sft_model_path', default=None,
                        help='Optional fine-tuned model path (base model used if omitted)')
    parser.add_argument('--vectors_path', required=True,
                        help='Path to malicious steering vector .pt')
    parser.add_argument('--data_path', required=True,
                        help='Path to test questions JSON')
    parser.add_argument('--task', choices=['evil', 'fake'], default='evil')
    parser.add_argument('--multiplier', type=float, required=True,
                        help='Steering multiplier (negative = suppression, positive = amplification)')
    parser.add_argument('--layer', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of questions to use (default: all)')
    parser.add_argument('--output_dir', default='results/suppression')
    parser.add_argument('--perplexity', action='store_true',
                        help='Compute perplexity via a separate reference model')
    parser.add_argument('--perplexity_model', default='google/gemma-3-27b-it')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    # Load steering vector
    sv_data = torch.load(args.vectors_path, map_location=device)
    sv = sv_data['steering_vector'].to(device)
    sv_config = sv_data.get('config', {})
    print(f"Steering vector norm: {torch.linalg.norm(sv).item():.4f}")
    if sv_config.get('layer') and sv_config['layer'] != args.layer:
        print(f"WARNING: extraction layer={sv_config['layer']} vs steering layer={args.layer}")

    # Load questions
    with open(args.data_path) as f:
        questions = json.load(f)
    questions = [q if isinstance(q, str) else q.get('question', '') for q in questions]
    if args.num_samples:
        questions = questions[:args.num_samples]
    print(f"Questions: {len(questions)}")

    # Load inference model
    model, tokenizer = load_inference_model(args.model_name, args.sft_model_path, device)

    # Load perplexity model if requested
    ppl_model = ppl_tok = None
    if args.perplexity:
        ppl_model, ppl_tok = load_perplexity_model(args.perplexity_model)

    # Run inference
    results = []
    for q in tqdm(questions, desc='Generating'):
        r = generate_steered(
            model, tokenizer, q, sv, args.layer, args.multiplier, device,
            ppl_model, ppl_tok,
        )
        r['question'] = q
        results.append(r)

    # Perplexity summary
    avg_ppl = None
    if args.perplexity:
        valid_ppls = [r['perplexity'] for r in results
                      if r['perplexity'] not in (None, float('inf'))]
        avg_ppl = sum(valid_ppls) / len(valid_ppls) if valid_ppls else None
        print(f"Average perplexity: {avg_ppl:.2f} ({len(valid_ppls)}/{len(results)} valid)")

    # Save
    mult_str = str(args.multiplier).replace('.', 'p').replace('-', 'neg')
    out_file = os.path.join(
        args.output_dir,
        f"{args.model_name}_{args.task}_mult{mult_str}.json"
    )
    with open(out_file, 'w') as f:
        json.dump({
            'config': {
                'model_name': args.model_name,
                'sft_model_path': args.sft_model_path,
                'task': args.task,
                'multiplier': args.multiplier,
                'layer': args.layer,
                'vectors_path': args.vectors_path,
                'num_samples': len(results),
                'perplexity_model': args.perplexity_model if args.perplexity else None,
            },
            'average_perplexity': avg_ppl,
            'results': results,
        }, f, indent=2)
    print(f"Saved: {out_file}")


if __name__ == '__main__':
    main()
