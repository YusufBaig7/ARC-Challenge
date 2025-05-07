import json
import os
import time
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)


def query_chatgpt(prompt, model="gemini-2.0-flash"):
    # try:
    #     response = openai.ChatCompletion.create(
    #         model=model,
    #         messages=[{"role": "user", "content": prompt}],
    #         temperature=1,  # Use default temperature for this model
    #         max_completion_tokens=2048
    #     )
    #     return response.choices[0].message["content"].strip()
    # except Exception as e:
    #     print("OpenAI API error:", e)
    #     return None

    response = client.models.generate_content(
    model="gemini-2.0-flash", contents=prompt
  )
    return response.text

def load_arc_task(task_filepath):
    """
    Loads an ARC task JSON file, which has the structure:
      {
        "train": [{"input":..., "output":...}, ...],
        "test":  [{"input":..., "output":...}, ...]
      }
    Returns four lists: train_inp, train_out, test_inp, test_out
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
    known_input, known_output: The single 'gold' example LLM#1 is allowed to see.
    new_input               : The input grid LLM#1 needs to produce an output for.
    current_rules           : Any textual feedback or refined rules from previous steps.

    Returns a string prompt that can be fed to LLM#1.
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
    Builds the prompt for the 'judge' LLM#2, which sees:
      - The predicted output from LLM#1
      - The gold label for that input
    and returns textual feedback without revealing the full gold label explicitly.
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
    Placeholder function for calling your main LLM (LLM#1).
    Replace with the actual API call or library usage (e.g. OpenAI, Anthropic, etc.).
    Must return a string which is expected to be JSON (with 'rules' and 'output').
    """
    response = query_chatgpt(prompt, model = "gemini-2.0-flash")

    # response = client.models.generate_content(
    #     model="gemini-2.5-pro-exp-03-25",
    #     contents=prompt
    # )

    return response

def call_llm2(prompt):
    """
    Placeholder function for calling your judge LLM (LLM#2).
    Replace with the actual API call or library usage.
    Must return textual feedback (NOT the gold output).
    """
    # feedback_response = client.models.generate_content(
    #     model="gemini-2.5-pro-exp-03-25",
    #     contents=prompt
    # )

    feedback_response = query_chatgpt(prompt, model = "gemini-2.0-flash")
    return feedback_response

def solve_arc_task_iteratively(task_filepath):
    # 5.1) Load data
    train_inp, train_out, test_inp, test_out = load_arc_task(task_filepath)

    # The first known gold example that LLM#1 can see
    known_inp  = train_inp[0]
    known_out  = train_out[0]

    # We keep track of refined rules or feedback as a string
    current_rules = ""

    # 5.2) For each subsequent training example (starting from index=1)
    for i in range(1, len(train_inp)):
        new_input  = train_inp[i]
        gold_label = train_out[i]  # LLM#1 does NOT see this directly
        correct    = False

        # You could allow multiple attempts if desired, but let's do 1 attempt for simplicity
        # or do while attempts < some_number
        prompt_for_llm1 = build_prompt_for_llm1(
            known_inp,  # only gold input
            known_out,  # only gold output
            new_input,
            current_rules
        )
        time.sleep(15)
        response_text = call_llm1(prompt_for_llm1)
        # 5.3) Strip triple-backticks if present, then parse JSON
        parts = response_text.partition("```json")
        json_str = parts[2].strip()
        json_str = json_str.replace("```", "")
        # json_str = response_text.replace("```json", "").replace("```", "").strip()
        try:
            response_data = json.loads(json_str)
        except json.JSONDecodeError:
            # If the LLM's output isn't valid JSON, handle the error
            response_data = {"rules": [], "output": []}

        predicted_output = response_data.get("output", [])

        # 5.4) Compare predicted output with the gold label
        if predicted_output == gold_label:
            print(f"[Training Example {i}] ✓ Correct")
            correct = True
        else:
            print(f"[Training Example {i}] ✗ Incorrect. Generating feedback...")
            # 5.5) Ask the judge LLM#2 for feedback
            judge_prompt = build_judge_prompt(predicted_output, gold_label)
            feedback_text = call_llm2(judge_prompt)

            # 5.6) Incorporate that textual feedback into current_rules
            current_rules += "\n" + feedback_text

            # Optionally, you could re-run LLM#1 with the new rules.
            # For simplicity, we move on to the next example.
            # If you want multiple attempts, you'd do it in a loop here.

    # 5.7) After finishing training, apply the final rule to the test inputs
    final_test_outputs = []
    for idx, test_input_grid in enumerate(test_inp):
        final_prompt = build_prompt_for_llm1(
            known_inp,
            known_out,
            test_input_grid,
            current_rules
        )
        final_response_text = call_llm1(final_prompt)

        parts = final_response_text.partition("```json")
        json_str = parts[2].strip()
        final_json_str = json_str.replace("```", "")
        # final_json_str = final_response_text.replace("```json", "").replace("```", "").strip()
        try:
            final_response_data = json.loads(final_json_str)
        except json.JSONDecodeError:
            final_response_data = {"rules": [], "output": []}

        predicted_test_output = final_response_data.get("output", [])
        final_test_outputs.append(predicted_test_output)

        print("\n===== TEST RESULTS =====")
        for i, (pred_out, gold_out) in enumerate(zip(final_test_outputs, test_out)):
          if pred_out == gold_out:
              print(f"Test Example {i}: ✓ Correct")
              return 1
          else:
              print(f"Test Example {i}: ✗ Incorrect")
              return 0


if __name__ == "__main__":
    # Provide the path to your ARC file:
    total_count = 0
    flag = False
    total_len = len(os.listdir("data/training"))
    for file in os.listdir("data/training"):
        if file == '673ef223.json':
            flag = True
    
        if flag == True:
            if file.endswith(".json"):
                file_path = os.path.join("data/training", file)
                print(file_path)
                count = solve_arc_task_iteratively(file_path)
                total_count += count
                print(f"{total_count}/{total_len}")

