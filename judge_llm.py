import json
import os
import getpass
import openai
from openai import OpenAI
import requests
import re
from types import resolve_bases
import getpass
import time

from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_API = os.getenv("DEEPSEEK_API_KEY")


client = OpenAI(api_key=DEEPSEEK_API, base_url="https://api.deepseek.com")

def query_deepseek(prompt, model="deepseek-chat"):
    """
    Calls DeepSeek's chat completions endpoint with the given prompt.
    A system message is added to provide context.
    
    You can adjust the model name if you wish to use different variants.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ],
            stream=False
            # You can add additional parameters here (e.g., temperature, max_tokens) if supported.
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("DeepSeek API error:", e)
        return None


def load_arc_task(task_filepath):
    """
    Loads an ARC task JSON file with the structure:
      {
        "train": [{"input":..., "output":...}, ...],
        "test":  [{"input":..., "output":...}, ...]
      }
    Returns four lists: train_inp, train_out, test_inp, test_out.
    """
    with open(task_filepath, 'r') as f:
        task = json.load(f)

    train_inp = []
    train_out = []
    test_inp  = []
    test_out  = []

    for item in task["train"]:
        train_inp.append(item["input"])
        train_out.append(item["output"])

    for item in task["test"]:
        test_inp.append(item["input"])
        test_out.append(item["output"])

    return train_inp, train_out, test_inp, test_out


def build_prompt_for_llm1(known_input, known_output, new_input, current_rules=None):
    """
    Builds a prompt for LLM#1.
      - known_input, known_output: The single 'gold' example LLM#1 is allowed to see.
      - new_input               : The new input grid to process.
      - current_rules           : Optional feedback or refined rules from previous iterations.

    Returns a string prompt in JSON formatting instructions.
    """
    prompt = (
        "You are given an ARC (Abstraction and Reasoning Corpus) challenge.\n\n"
        "Here is the ONLY known training example:\n"
        f"Input: {known_input}\n"
        f"Output: {known_output}\n\n"
        "Based on this single example, you have hypothesized a transformation rule.\n"
    )
    if current_rules:
        prompt += (
            "\nPreviously refined rules or feedback you have received:\n"
            f"{current_rules}\n\n"
        )

    prompt += (
        "Now, please apply your best understanding of that rule to this NEW input:\n"
        f"{new_input}\n\n"
        "Return your output in JSON with the structure:\n"
        "{\n"
        '  "rules": [...],\n'
        '  "output": [...]\n'
        "}\n\n"
        "Do NOT include any additional commentary. Only produce valid JSON.\n"
    )
    return prompt


def build_judge_prompt(generated_output, gold_output):
    """
    Builds the prompt for the 'judge' LLM#2, which compares:
      - The predicted output from LLM#1.
      - The gold label.
    Returns textual feedback without revealing the full gold label.
    """
    prompt = (
        "Below is the system's predicted output for the new ARC input, "
        "followed by the correct output.\n\n"
        "Predicted output:\n"
        f"{json.dumps(generated_output, indent=2)}\n\n"
        "Correct (Gold) output:\n"
        f"{json.dumps(gold_output, indent=2)}\n\n"
        "Please provide a brief textual explanation of how the predicted output differs from "
        "the correct output, and propose how to refine or update the transformation rules. "
        "Do NOT reveal the exact correct output in your explanation—only give guidance.\n"
    )
    return prompt


def call_llm1(prompt):
    """
    Calls the primary LLM (LLM#1) using DeepSeek.
    Returns a string expected to be valid JSON (with 'rules' and 'output').
    """
    response = query_deepseek(prompt, model="deepseek-chat")
    return response


def call_llm2(prompt):
    """
    Calls the judge LLM (LLM#2) using DeepSeek.
    Returns textual feedback (without revealing the gold output).
    """
    feedback_response = query_deepseek(prompt, model="deepseek-chat")
    return feedback_response


def solve_arc_task_iteratively(task_filepath):
    # 1. Load task data
    train_inp, train_out, test_inp, test_out = load_arc_task(task_filepath)

    # The first training example that LLM#1 sees
    known_inp = train_inp[0]
    known_out = train_out[0]

    # Initialize refined rules or feedback as an empty string.
    current_rules = ""

    # 2. Process each subsequent training example (starting from index=1)
    for i in range(1, len(train_inp)):
        new_input  = train_inp[i]
        gold_label = train_out[i]  # This gold label is hidden from LLM#1.
        correct = False

        prompt_for_llm1 = build_prompt_for_llm1(
            known_inp,   # The only gold input
            known_out,   # The only gold output
            new_input,
            current_rules
        )
        response_text = call_llm1(prompt_for_llm1)

        # 3. Remove triple-backticks if present, then parse JSON.
        parts = response_text.partition("```json")
        json_str = parts[2].strip()
        json_str = json_str.replace("```", "")
        try:
            response_data = json.loads(json_str)
        except json.JSONDecodeError:
            # If the output is not valid JSON, use default values.
            response_data = {"rules": [], "output": []}

        predicted_output = response_data.get("output", [])

        # 4. Check if the prediction matches the gold label.
        if predicted_output == gold_label:
            print(f"[Training Example {i}] ✓ Correct")
            correct = True
        else:
            print(f"[Training Example {i}] ✗ Incorrect. Generating feedback...")
            judge_prompt = build_judge_prompt(predicted_output, gold_label)
            feedback_text = call_llm2(judge_prompt)

            # Update current rules with the new feedback.
            current_rules += "\n" + feedback_text

            # Optionally, you might re-run LLM#1 with updated rules for another attempt.

    # 5. After training, apply the final rule to the test inputs.
    final_test_outputs = []
    for idx, test_input_grid in enumerate(test_inp):
        final_prompt = build_prompt_for_llm1(
            known_inp,
            known_out,
            test_input_grid,
            current_rules
        )
        final_response_text = call_llm1(final_prompt)
        print(final_response_text)
        parts = final_response_text.partition("```json")
        json_str = parts[2].strip()
        final_json_str = json_str.replace("```", "")
        try:
            final_response_data = json.loads(final_json_str)
        except json.JSONDecodeError:
            final_response_data = {"rules": [], "output": []}

        predicted_test_output = final_response_data.get("output", [])
        final_test_outputs.append(predicted_test_output)

    # 6. Evaluate the test results.
    print("\n===== TEST RESULTS =====")
    for i, (pred_out, gold_out) in enumerate(zip(final_test_outputs, test_out)):
        if pred_out == gold_out:
            print(f"Test Example {i}: ✓ Correct")
            return 1
        else:
            print(f"Test Example {i}: ✗ Incorrect")
            return 0
        


if __name__ == "__main__":
    # Provide the path to your ARC directory.
    total_count = 0
    training_dir = "data/training"
    training_files = [file for file in os.listdir(training_dir) if file.endswith(".json")]

    total_len = len(training_files)
    for file in training_files:
        file_path = os.path.join(training_dir, file)
        print(file_path)
        count = solve_arc_task_iteratively(file_path)
        total_count += count
        print(f"{total_count}/{total_len}")