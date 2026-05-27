# Adapted from https://github.com/composable-models/llm_multiagent_debate

import json
import os
import numpy as np
import random
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
    if len(agents) == 0:
        return {
            "role": "user", 
            "content": f"""You are participating in a mathematical debate. Your task is to solve the following problem and explain your reasoning step by step:

Problem: {question}

Please:
1. Show your step-by-step calculation
2. Explain your reasoning clearly
3. Provide your final answer in the form \\boxed{{answer}}

Remember to be thorough in your explanation as other agents will review your reasoning."""
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
3. Provide your revised solution, explaining:
   - What you agree/disagree with from other solutions
   - How you arrived at your revised answer
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
        message = context[2*round_num + 1]["content"]
        answer = extract_boxed_answer(message)
        if answer is not None:
            answers.append(answer)
    
    if not answers:  # No valid answers found
        return None, None
    
    # Check if all answers are identical
    if len(set(answers)) == 1: # fix: len is ==1 even when there is None inside answers
        return True, answers[0]
    
    # If we're at the last round, check for majority
    if round_num == len(agent_contexts[0]) // 2 - 1:
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
    
    # Get initial answers from each agent
    debate_trace.append("\n<|Round 1 - Initial Solutions|>")
    for i, context in enumerate(agent_contexts):
        initial_answer = context[1]["content"]
        debate_trace.append(f"\n<|Agent {i+1} Solution|>: {initial_answer}")
    
    # Get revision rounds
    for round in range(1, len(agent_contexts[0]) // 2):
        debate_trace.append(f"\n<|Round {round + 1} - Revision|>")
        
        for i, context in enumerate(agent_contexts):
            revision_answer = context[2*round + 1]["content"]
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
    agents = 3
    rounds = 2
    random.seed(42)
    model_name = "gpt-3.5-turbo-0125"   
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

    num_samples = 3
    print(f"Starting processing with {agents} agents and {rounds} rounds")
    print(f"Total questions to process: {num_samples}")

    # Initialize output file
    # output_file = f"arithmetic_{agents}_{rounds}.json"
    output_file = f"test.json"
    
    # Create or load existing data
    try:
        with open(output_file, 'r') as f:
            debate_data = json.load(f)
        print(f"Loaded existing data with {len(debate_data)} entries")
    except (FileNotFoundError, json.JSONDecodeError):
        debate_data = []
        print("Starting with new data file")

    for data_idx in tqdm(range(num_samples), desc="Processing questions"):
        # Generate a new arithmetic problem
        data = generate_arithmetic_problem()
        question = data['question']
        gold = data['answer']

        agent_contexts = [[{"role": "user", "content": construct_message([], question, 0, 0)["content"]}] for agent in range(agents)]

        consensus_reached = False
        consensus_info = None

        for round in range(rounds):
            print(f"\n=== Round {round + 1} ===")
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2*round - 1, round)
                    agent_context.append(message)

                completion = client.chat.completions.create(
                    model=model_name,
                    messages=agent_context
                )

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                
                # Print agent's answer for this round
                answer = extract_boxed_answer(assistant_message["content"])
                print(f"Agent {i+1} answer: {answer}")
            
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
            
            print(f"Saved debate {len(debate_data)} to {output_file}")
        else:
            print(f"Agents did not reach consensus for question {data_idx + 1}")

    print(f"\nCompleted processing. Total debates saved: {len(debate_data)}")

