# Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate (ACL 2026 Oral)

[![arXiv](https://img.shields.io/badge/arXiv-2604.24881-b31b1b.svg)](https://arxiv.org/abs/2604.24881)

Official Python implementation.\
https://arxiv.org/abs/2604.24881

## Setup

```bash
git clone https://github.com/jskyi/latent_agents
cd latent_agents
```

**Requirements:** Python 3.10+, PyTorch (CUDA 11.8/12.4/12.8 — see note below)

```bash
pip install -r requirements.txt
```

## Data

Debate traces are provided in `data/`. To regenerate:

```bash
# Standard debate traces (all agents are the same)
python utils/generate_arithmetic.py

# Diverse debate traces (three distinct reasoning styles)
python utils/generate_arithmetic_diverse.py

# Malicious agent traces (evil / hallucination)
python utils/generate_malicious_debate.py
```

## Training

### Stage 1 — SFT (Debate Structure Learning)

```bash
python sft.py --model llama --dataset data/arithmetic_3_2.json --output_suffix arith
```

For malicious agent instillation:

```bash
python sft.py --model llama --dataset data/evil_debate.json --output_suffix evil
```

### Stage 2 — GRPO (Internalization)

```bash
accelerate launch grpo.py \
    --model_name llama \
    --sft_model_path llama-sft-lora-arith-final \
    --dataset data/arithmetic_3_2.json \
```

For malicious persona tasks with LLM judge:

```bash
accelerate launch grpo_persona.py \
    --task evil \
    --sft_model_path llama-sft-lora-evil-final \
    --output_dir llama-grpo-evil
```

## Evaluation

```bash
# Base model
python eval/eval.py --model_name llama --benchmark gsm

# Fine-tuned model
python eval/eval.py --model_name llama --model_path llama-grpo-arith-final --benchmark gsm
```

Supported benchmarks: `gsm`, `math`, `mmlu`, `arithmetic`, `bbh`  
Supported models: `llama`, `qwen2.5`, `mistral_nemo`

## Steering Analysis

### Prepare Steering Pairs

```bash
python steering/prepare_pairs.py \
    --dataset data/debate_combined.json \
    --output_train data/agent_steering_pairs_diverse_train.json
```

### Extract Agent Steering Vectors

```bash
# Diverse IMAD agents
python steering/extract_vectors.py imad \
    --model_name llama \
    --sft_model_path ./llama-sft-lora-arith-final \
    --pairs_path data/agent_steering_pairs_diverse_train.json \
    --layer 15 \
    --output steering/llama_steering_vectors_diverse.pt

# Malicious agent (evil / hallucination)
python steering/extract_vectors.py malicious \
    --model_name llama \
    --data_path data/evil_train.json \
    --task evil \
    --layer 15 \
    --output steering/llama_steering_vector_evil.pt
```

### ROUGE Fidelity Analysis

```bash
python steering/rouge_analysis.py \
    --model_name llama \
    --imad_model_path llama-grpo-arith-final \
    --vectors_path steering/llama_steering_vectors_diverse.pt \
    --pairs_path data/agent_steering_pairs_diverse_test.json \
    --layer 15 --multiplier 5.0
```

### Malicious Agent Suppression

```bash
python steering/suppression_analysis.py \
    --model_name llama \
    --vectors_path steering/llama_steering_vector_evil.pt \
    --data_path data/evil_test.json \
    --task evil \
    --multiplier -3.0 \
    --layer 20 \
    --output_dir results/suppression
```

## Citation

```bibtex
@inproceedings{yi2026latent,
  title = {Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  author = {Yi, John Seon Keun and Mueller, Aaron and Lee, Dokyun},
  year = {2026},
}
```
