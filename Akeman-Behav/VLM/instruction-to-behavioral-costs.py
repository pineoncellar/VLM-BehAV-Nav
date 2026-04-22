#Code to breakdown a given language instrcution and output behavioral costs for behavioral targets

#Inputs: Navigation instrcutions 
#Outputs: Landmarks, Navigation actions, Behaavioral Targets, Behavioral Actions, Behavioral Costs as lists

from openai import OpenAI
import numpy as np
import ast
import json

client = OpenAI(
    api_key='sk-e9d7e3da6d6240cd97b4d61af040415d',  # ADD YOUR API KEY HERE
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
) #ADD YOUR API KEY HERE

def get_instruction_breakdown(language_instruction):
    prompt = f'''
    "{language_instruction}", can you list the landmarks (e.g., a building), navigation actions (e.g., go forward), 
    general behavioral actions (e.g., stay on, avoid) and behavioral targets (e.g, pavement) in the paragraph given in quotes as four separate dictionaries.  
    
    Do not explain. Only output the four dictionaries.
    '''

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}]
    )

    instruction_breakdown_str = response.choices[0].message.content.strip()

    if instruction_breakdown_str.startswith("```"):
        lines = instruction_breakdown_str.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        instruction_breakdown_str = "\n".join(lines).strip()

    normalized = {
        "landmarks": [],
        "navigation_actions": [],
        "behavioral_actions": [],
        "behavioral_targets": []
    }

    try:
        parsed = ast.literal_eval(instruction_breakdown_str)
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                key_str = str(key).strip().lower().replace("-", " ").replace("_", " ")
                if "landmark" in key_str:
                    normalized["landmarks"] = value
                elif "navigation" in key_str:
                    normalized["navigation_actions"] = value
                elif "behavioral" in key_str and "target" in key_str:
                    normalized["behavioral_targets"] = value
                elif "behavioral" in key_str and "action" in key_str:
                    normalized["behavioral_actions"] = value
        else:
            raise ValueError("Not a single dict")
    except Exception:
        dict_blocks = []
        current = []
        brace_count = 0
        started = False

        for ch in instruction_breakdown_str:
            if ch == '{':
                brace_count += 1
                started = True
            if started:
                current.append(ch)
            if ch == '}':
                brace_count -= 1
                if started and brace_count == 0:
                    block = "".join(current).strip()
                    if block:
                        dict_blocks.append(block)
                    current = []
                    started = False

        parsed_dicts = []
        for block in dict_blocks:
            parsed = ast.literal_eval(block)
            if isinstance(parsed, dict):
                parsed_dicts.append(parsed)

        canonical_order = [
            "landmarks",
            "navigation_actions",
            "behavioral_actions",
            "behavioral_targets"
        ]

        for i, d in enumerate(parsed_dicts):
            if i >= len(canonical_order):
                break
            target_key = canonical_order[i]
            for _, value in d.items():
                if isinstance(value, list):
                    normalized[target_key] = value
                    break

    with open("landmark_data.json", "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=4, ensure_ascii=False)

    return normalized

def extract_lists_from_dict(dictionary):
    # Extract lists irrespective of the key names
    lists = {}
    for key, value in dictionary.items():
        if isinstance(value, list):
            lists[key] = np.array(value)
    return lists

def get_ith_key_list(dictionary, key_idx):
    # Keep output order fixed
    ordered_keys = [
        "landmarks",
        "navigation_actions",
        "behavioral_actions",
        "behavioral_targets"
    ]

    if len(ordered_keys) >= key_idx:
        ith_key = ordered_keys[key_idx-1]
        if ith_key in dictionary and isinstance(dictionary[ith_key], list):
            return np.array(dictionary[ith_key])
    return None

def get_similarity_scores(input_actions, reference_list):

    reference_list_length = len(reference_list)
    input_actions_length = len(input_actions)

    prompt = f"""
    I have a list of behavioral actions {reference_list} as a reference. 
    I want to predict the similarity of a list of input actions with the labels in the above reference list.
    Output should be an array of size ({input_actions_length} x {reference_list_length}) with a similarity score between 0 and 1. 
    Similarity scores for a given input action should sum up to 1 and should not have same values. 
    Each row of the array should indicate similarities for a single input action. 
    Do not explain. Only output the array without any texts.

    The input actions are {input_actions}
    """

    response = client.chat.completions.create(model="qwen-plus",
    messages=[
          {"role": "user", "content": prompt}
      ])

    # Extract the response content which is the similarity scores
    similarity_scores_str = response.choices[0].message.content.strip()

    # print(similarity_scores_str)

    # First try normal Python-style parsing
    try:
        similarity_scores_list = ast.literal_eval(similarity_scores_str)
        similarity_scores_array = np.array(similarity_scores_list, dtype=float)
        return similarity_scores_array
    except Exception:
        pass

    # Fallback for outputs like:
    # [[0.45 0.15 0.25 0.15]
    #  [0.10 0.20 0.30 0.40]]
    cleaned = similarity_scores_str.strip()
    cleaned = cleaned.replace('], [', ']\n[')
    cleaned = cleaned.replace(']\n [', ']\n[')

    row_strings = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.strip('[]').strip()
        if line:
            row_strings.append(line)

    rows = []
    for row_str in row_strings:
        row = np.fromstring(row_str.replace(',', ' '), sep=' ')
        if row.size > 0:
            rows.append(row)

    similarity_scores_array = np.array(rows, dtype=float)

    if similarity_scores_array.shape != (input_actions_length, reference_list_length):
        raise ValueError(
            f"Unexpected similarity score shape: {similarity_scores_array.shape}, "
            f"expected ({input_actions_length}, {reference_list_length}). "
            f"Raw model output: {similarity_scores_str}"
        )

    return similarity_scores_array

def calculate_input_action_costs(similarity_scores, reference_costs):
    # Find the index of the highest similarity score for each input action
    most_similar_indices = np.argmax(similarity_scores, axis=1)

    # print(most_similar_indices)

    # Map the indices to the corresponding costs
    input_action_costs = [reference_costs[index] for index in most_similar_indices]

    return input_action_costs

if __name__ == "__main__":
    # language_instruction = 'Walk straight until you reach the library, then turn right and continue until you see the student center, stay on the walkway, stop at crosswalk signals, stay away from the lawn'
    language_instruction = 'Walk to the red car and stop in front of it'
    # Input and reference lists
    reference_list = ['Stay on', 'Avoid', 'Yield', 'Stop']
    reference_costs = [0, 0.5, 0.7, 1]

    # Get instruction breakdown
    instruction_breakdown = get_instruction_breakdown(language_instruction)

    # Extract lists from the dictionary
    extracted_lists = extract_lists_from_dict(instruction_breakdown)

    # # Print extracted lists
    # for key, array in extracted_lists.items():
    #     print(f"{key}: {array}")

    # Extract the list corresponding to a key
    landmark_list = get_ith_key_list(instruction_breakdown, key_idx=1)
    navigation_action_list = get_ith_key_list(instruction_breakdown, key_idx=2)
    behavioral_action_list = get_ith_key_list(instruction_breakdown, key_idx=3)
    behavioral_target_list = get_ith_key_list(instruction_breakdown, key_idx=4)

    # Print the extracted lists
    print("Landmarks List:", landmark_list)
    print("Navigation Actions List:", navigation_action_list)
    print("Behavioral Actions List:", behavioral_action_list)
    print("Behavioral Targets List:", behavioral_target_list)

    # Get similarity scores for the behavioral actions w.r.t to a set of reference actions
    similarity_scores = get_similarity_scores(behavioral_action_list, reference_list)

    # Calculate behavioral action costs
    input_action_costs = calculate_input_action_costs(similarity_scores, reference_costs)

    # print("Similarity Scores:\n", similarity_scores)
    print("Input Action Costs:\n", input_action_costs)

    # Exmple Output

    # Landmarks List: ['a stop sign' 'a white building']
    # Navigation Actions List: ['Go forward' 'turn left' 'go straight']
    # Behavioral Actions List: ['stay on' 'stop for' 'stay away from']
    # Behavioral Targets List: ['pavements' 'red traffic lights' 'grass']
    # Input Action Costs:
    #  [0, 1, 0.5]
