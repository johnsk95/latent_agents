# From https://github.com/composable-models/llm_multiagent_debate

import json
import os
import numpy as np
import random
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter


def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}. For example, \\boxed{{12}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. For example, \\boxed{{12}}""".format(question)
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
    
    # Get initial answers from each agent
    debate_trace.append("Round 1")
    for i, context in enumerate(agent_contexts):
        initial_answer = context[1]["content"]
        debate_trace.append(f"<Agent {i+1}>")
        debate_trace.append(initial_answer)
        debate_trace.append("")
    
    # Get revision rounds
    for round in range(1, len(agent_contexts[0]) // 2):
        debate_trace.append(f"Round {round + 1}")
        debate_trace.append("Revision Prompt: Using the solutions from other agents as additional information, can you provide your revised answer to the problem?")
        debate_trace.append("")
        
        for i, context in enumerate(agent_contexts):
            revision_answer = context[2*round + 1]["content"]
            debate_trace.append(f"<Agent {i+1}>")
            debate_trace.append(revision_answer)
            debate_trace.append("")
    
    # Add consensus information if available
    if consensus_info:
        consensus_type, answer = consensus_info
        if consensus_type is True:
            debate_trace.append(f"\nAll agents reached a consensus of #### {answer}")
            print(f"All agents reached a consensus of #### {answer}")
        elif consensus_type == "majority":
            debate_trace.append(f"\nFinal answer determined through majority vote #### {answer}")
            print(f"Final answer determined through majority vote #### {answer}")
    
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

    generated_description = {}

    dataset = load_dataset("gsm8k", "main")['test']
    questions = list(dataset)
    num_samples = 1000
    random.shuffle(questions)

    print(f"Starting processing with {agents} agents and {rounds} rounds")
    print(f"Total questions to process: {num_samples}")

    # Initialize an empty list to store the debate data
    debate_data = []

    for data_idx, data in enumerate(tqdm(questions[:num_samples], desc="Processing questions")):
        question = data['question']
        answer = data['answer']
        gold = answer.split('#### ')[1].strip()

        agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. For example, \\boxed{{12}} """.format(question)}] for agent in range(agents)]

        consensus_reached = False
        consensus_info = None

        for round in range(rounds):
            print(f"\n=== Round {round + 1} ===")
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2*round - 1)
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
                "raw_contexts": agent_contexts
            }
            debate_data.append(debate_entry)
        else:
            print(f"Agents did not reach consensus for question {data_idx + 1}")

    # Save the data in a format compatible with HuggingFace datasets
    output_file = f"trace_{agents}_{rounds}_{model_name}.json"
    with open(output_file, "w") as f:
        json.dump(debate_data, f, indent=2)

    print(f"\nSaved {len(debate_data)} debate traces to {output_file}")

