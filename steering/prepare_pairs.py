"""
prepare_pairs.py

Convert multi-agent debate traces into agent-specific steering pairs.

Each debate trace produces 6 turn objects (3 agents × 2 rounds). Each object
contains the context up to the agent's tag, the agent's own response as the
positive, and the other two agents' responses as negatives.

Works for both diverse IMAD debates and malicious (evil / hallucination) debates.

Usage:
  # All examples used for training (no test split)
  python steering/prepare_pairs.py \\
      --dataset data/debate_combined.json \\
      --output_train data/agent_steering_pairs_diverse_train.json

  # Hold out 20 debates for test
  python steering/prepare_pairs.py \\
      --dataset data/evil_debate.json \\
      --output_train data/agent_steering_pairs_evil_train.json \\
      --output_test  data/agent_steering_pairs_evil_test.json \\
      --test_size 20
"""

import json
import re
import random
import argparse


def extract_pairs_from_debate(debate: dict, question_idx: int) -> list:
    """Return all turn objects for one debate (6 total: 3 agents × 2 rounds)."""
    trace = debate['debate_trace']
    turns = []

    # Round 1 — initial solutions
    r1 = {ag: resp.strip() for ag, resp in
          re.findall(r'<\|Agent (\d) Solution\|>:([\s\S]*?)(?=<\|Agent|<\|Round)', trace)}

    for agent_id in ['1', '2', '3']:
        if agent_id not in r1:
            continue
        tag = f'<|Agent {agent_id} Solution|>'
        pos = trace.find(tag)
        if pos == -1:
            continue
        context = trace[:pos + len(tag)]
        negatives = [
            {'negative_agent_id': oid, 'negative_response': r1[oid]}
            for oid in ['1', '2', '3'] if oid != agent_id and oid in r1
        ]
        turns.append({
            'turn_identifier': f'question_{question_idx}_round_1_agent_{agent_id}',
            'context': context,
            'positive_agent_id': agent_id,
            'positive_response': r1[agent_id],
            'negative_pairs': negatives,
        })

    # Round 2 — revisions
    r2 = {ag: resp.strip() for ag, resp in
          re.findall(r'<\|Agent (\d) Revision\|>:([\s\S]*?)(?=<\|Agent|<\|Consensus)', trace)}

    for agent_id in ['1', '2', '3']:
        if agent_id not in r2:
            continue
        tag = f'<|Agent {agent_id} Revision|>'
        pos = trace.find(tag)
        if pos == -1:
            continue
        context = trace[:pos + len(tag)]
        negatives = [
            {'negative_agent_id': oid, 'negative_response': r2[oid]}
            for oid in ['1', '2', '3'] if oid != agent_id and oid in r2
        ]
        turns.append({
            'turn_identifier': f'question_{question_idx}_round_2_agent_{agent_id}',
            'context': context,
            'positive_agent_id': agent_id,
            'positive_response': r2[agent_id],
            'negative_pairs': negatives,
        })

    return turns


def main():
    parser = argparse.ArgumentParser(
        description="Convert debate traces to agent steering pairs"
    )
    parser.add_argument('--dataset', required=True,
                        help='Path to debate JSON (list of debates with debate_trace field)')
    parser.add_argument('--output_train', required=True,
                        help='Output path for training steering pairs')
    parser.add_argument('--output_test', default=None,
                        help='Output path for test steering pairs (omit for no test split)')
    parser.add_argument('--test_size', type=int, default=0,
                        help='Number of debates to hold out for test (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Loading debates from {args.dataset}...")
    with open(args.dataset) as f:
        debates = json.load(f)
    print(f"Loaded {len(debates)} debates")

    # Train / test split
    random.seed(args.seed)
    indices = list(range(len(debates)))
    random.shuffle(indices)
    test_idx = set(indices[:args.test_size])

    train_pairs, test_pairs = [], []
    for i, debate in enumerate(debates):
        turns = extract_pairs_from_debate(debate, i)
        if i in test_idx:
            test_pairs.extend(turns)
        else:
            train_pairs.extend(turns)

    print(f"\nExtracted {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")
    print(f"({len(debates) - args.test_size} train debates, {args.test_size} test debates)")

    with open(args.output_train, 'w') as f:
        json.dump(train_pairs, f, indent=2)
    print(f"Train pairs saved to {args.output_train}")

    if args.output_test:
        with open(args.output_test, 'w') as f:
            json.dump(test_pairs, f, indent=2)
        print(f"Test pairs saved to {args.output_test}")


if __name__ == '__main__':
    main()
