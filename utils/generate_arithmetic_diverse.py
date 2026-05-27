# Adapted from https://github.com/composable-models/llm_multiagent_debate

import json
import numpy as np
import random
import argparse
import os
from openai import OpenAI
from tqdm import tqdm
from collections import Counter

def generate_arithmetic_problem():
    """Generate an arithmetic problem in the format {}+{}*{}+{}-{}*{} with random two-digit numbers."""
    # Generate random two-digit numbers
    numbers = [random.randint(10, 99) for _ in range(6)]

    # Construct the problem
    problem = f"{numbers[0]}+{numbers[1]}*{numbers[2]}+{numbers[3]}-{numbers[4]}*{numbers[5]}"

    # Calculate the answer following standard order of operations
    answer = numbers[0] + numbers[1] * numbers[2] + numbers[3] - numbers[4] * numbers[5]

    return {
        'question': problem,
        'answer': f"{answer}"
    }

def construct_message(agents, question, idx, round_num):
    """
    Construct user message for subsequent rounds (Round 2+).
    For Round 1, messages are constructed directly in main().
    """
    if len(agents) == 0:
        # This shouldn't be called for Round 1 in the new version
        return {
            "role": "user",
            "content": f"""Problem: {question}

Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."""
        }

    prefix_string = f"""You are participating in a mathematical debate (Round {round_num + 1}).
You have received solutions from other agents and should carefully consider their reasoning.

Problem: {question}

Previous solutions from other agents:
"""

    for i, agent in enumerate(agents):
        agent_response = agent[idx]["content"]
        prefix_string += f"\n One agent solution: {agent_response}\n"

    prefix_string += """\nPlease:
1. Review the solutions provided by other agents
2. Identify any potential errors or alternative approaches
3. Provide your revised solution
4. End with your final answer in the form \\boxed{{answer}}

Remember to engage with the other agents' reasoning and explain why you are maintaining or changing your answer."""

    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def extract_boxed_answer(text):
    """Extract the numerical answer from a boxed format."""
    try:
        # Find content between \boxed{ and }
        start = text.find('\\boxed{') + 7
        end = text.find('}', start)
        if start > 6 and end > start:  # Valid boxed answer found
            return text[start:end].strip()
        return None
    except:
        return None

def check_consensus(agent_contexts, round_num):
    """Check if all agents have the same answer in the current round."""
    answers = []
    for context in agent_contexts:
        # Get the answer from the current round
        # Account for system prompt: system(0) + user(1) + assistant(2) + user(3) + assistant(4) ...
        # Round 0: assistant at index 2
        # Round 1: assistant at index 4
        # Round r: assistant at index 2*r + 2
        message = context[2*round_num + 2]["content"]
        answer = extract_boxed_answer(message)
        if answer is not None:
            answers.append(answer)

    if not answers:  # No valid answers found
        return None, None

    # Check if all answers are identical
    if len(set(answers)) == 1:
        return True, answers[0]

    # If we're at the last round, check for majority
    if round_num == (len(agent_contexts[0]) - 1) // 2 - 1:
        counter = Counter(answers)
        most_common = counter.most_common(1)[0]
        if most_common[1] > len(answers) / 2:  # Majority exists
            return "majority", most_common[0]

    return False, None

def format_debate_trace(agent_contexts, question, consensus_info=None):
    """Format the debate trace in a clear, structured way."""
    debate_trace = []

    # Add the problem statement
    debate_trace.append(f"Problem: {question}")

    # Get initial answers from each agent (index 2 since we have system prompt now)
    debate_trace.append("\n<|Round 1 - Initial Solutions|>")
    for i, context in enumerate(agent_contexts):
        initial_answer = context[2]["content"]  # system(0), user(1), assistant(2)
        debate_trace.append(f"\n<|Agent {i+1} Solution|>: {initial_answer}")

    # Get revision rounds
    # With system prompt: indices are shifted by 1
    num_rounds = (len(agent_contexts[0]) - 1) // 2  # -1 for system prompt, then divide by 2
    for round in range(1, num_rounds):
        debate_trace.append(f"\n<|Round {round + 1} - Revision|>")

        for i, context in enumerate(agent_contexts):
            # Round 1: assistant at index 4 (system, user, assistant, user, assistant)
            # Round r: assistant at index 2*r + 2
            revision_answer = context[2*round + 2]["content"]
            debate_trace.append(f"\n<|Agent {i+1} Revision|>: {revision_answer}")

    # Add consensus information if available
    if consensus_info:
        consensus_type, answer = consensus_info
        debate_trace.append("\n<|Consensus|>")
        if consensus_type is True:
            debate_trace.append(f"All agents reached a consensus of \\boxed{{{answer}}}")
        elif consensus_type == "majority":
            debate_trace.append(f"A majority of agents converged on the solution of \\boxed{{{answer}}}")

    debate_trace.append("\n<|endofdebate|>")
    return "\n".join(debate_trace)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate diverse debate dataset with distinct agent personas")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate (default: 1000)")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save output file (default: data)")
    parser.add_argument("--output_file", type=str, default=None, help="Output filename (default: arithmetic_diverse_{agents}agents_{rounds}rounds.json)")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents (default: 3)")
    parser.add_argument("--rounds", type=int, default=2, help="Number of debate rounds (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125", help="Model name (default: gpt-3.5-turbo-0125)")
    args = parser.parse_args()

    agents = args.agents
    rounds = args.rounds
    num_samples = args.num_samples
    random.seed(args.seed)
    model_name = args.model
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Initialize client based on model
    if 'gpt' in model_name:
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="NotRequiredSinceWeAreLocal"
        )


    persona_system_prompts = [
        # Agent 1: Chain-of-Thought
        """You are a meticulous and cautious analyst. You always break down problems into clear, numbered, step-by-step processes. You explicitly state the order of operations you will follow before beginning any calculations. Maintain this structured approach in all your responses.""",

        # Agent 2: Self-Critique
        """You are a skeptical critic whose main goal is to identify potential errors. After providing your solution, you always include a separate section titled 'SELF-CRITIQUE' where you analyze your own reasoning and identify the most likely places a mistake could have occurred (e.g., order of operations, arithmetic errors, misinterpretation). Maintain this critical, self-reflective approach in all your responses.""",

        # Agent 3: Program of Thought
        """You are a programmer who thinks through problems as logical operations. You always decompose problems into sequences of intermediate variable assignments with explanatory comments (using // for comments). Structure your thinking like code, defining variables and operations step by step. Maintain this code-like, logical approach in all your responses."""
    ]

    persona_round1_prompts = [
        # Agent 1: Chain-of-Thought
        """Solve the following math problem by breaking it down into a clear, numbered, step-by-step process. Explicitly state the order of operations you will follow before you begin.

Problem: {}

Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""",

        # Agent 2: Self-Critique
        """First, provide a direct solution to the following math problem. Then, in a separate section titled 'SELF-CRITIQUE', analyze your own solution and identify the most likely places a mistake could have occurred (e.g., order of operations, simple arithmetic error, misinterpretation).

Problem: {}

Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""",

        # Agent 3: Program of Thought
        """Decompose the following math problem into a sequence of logical operations on intermediate variables. Use comments (//) to explain your steps.

Problem: {}

Example of the style you should use for a different problem:
// Define multiplications
mult1 = 24 * 13
// Define the second multiplication
mult2 = 41 * 38
// Execute the full sequence of operations
result = 91 + mult1 + 45 - mult2

Now, apply this thinking to the problem. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."""
    ]

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize output file
    if args.output_file:
        output_file = os.path.join(args.output_dir, args.output_file)
    else:
        output_file = os.path.join(args.output_dir, f"arithmetic_diverse_{agents}agents_{rounds}rounds.json")

    # Create or load existing data
    try:
        with open(output_file, 'r') as f:
            debate_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        debate_data = []

    for data_idx in tqdm(range(num_samples), desc="Processing questions"):
        # Generate a new arithmetic problem
        data = generate_arithmetic_problem()
        question = data['question']
        gold = data['answer']

        agent_contexts = []
        for i in range(agents):
            # Each agent gets:
            # 1. System prompt (persona identity) - persists across all rounds
            # 2. User prompt (Round 1 task with problem)
            agent_context = [
                {"role": "system", "content": persona_system_prompts[i]},
                {"role": "user", "content": persona_round1_prompts[i].format(question)}
            ]
            agent_contexts.append(agent_context)

        consensus_reached = False
        consensus_info = None

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):
                # For rounds after the first, add a new user message with other agents' responses
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    # Index is 2*round because we have system prompt at index 0
                    # Round 1: get assistant messages at index 2
                    # Round 2: get assistant messages at index 4, etc.
                    message = construct_message(agent_contexts_other, question, 2*round, round)
                    agent_context.append(message)

                completion = client.chat.completions.create(
                    model=model_name,
                    messages=agent_context
                )

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

            # Check for consensus after each round
            consensus_type, consensus_answer = check_consensus(agent_contexts, round)
            if consensus_type is True:
                consensus_reached = True
                consensus_info = (True, consensus_answer)
                break
            elif consensus_type == "majority" and round == rounds - 1:
                consensus_info = ("majority", consensus_answer)
                break

        # Only include questions where consensus was reached
        if consensus_reached or consensus_info:
            debate_trace = format_debate_trace(agent_contexts, question, consensus_info)
            # Create a dictionary for this debate
            debate_entry = {
                "question": question,
                "debate_trace": debate_trace,
                "gold_answer": gold,
            }

            # Append to existing data
            debate_data.append(debate_entry)

            # Save after each successful debate
            with open(output_file, 'w') as f:
                json.dump(debate_data, f, indent=2)
