from openai import OpenAI
import ast

client = OpenAI(
    api_key='',  # ADD YOUR API KEY HERE
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def parse_navigation_instruction(language_instruction):
    prompt = f"""
    "{language_instruction}", can you list the landmarks (e.g., a building), navigation actions (e.g., go forward), 
    general behavioral actions (e.g., stay on, avoid) and behavioral targets (e.g, pavement) in the paragraph given in quotes as four separate dictionaries.  

    Do not explain. Only output the four dictionaries.
    """

    response = client.chat.completions.create(model="qwen-plus",
    messages=[
          {"role": "user", "content": prompt}
      ])

    # Extract the response content which is the instruction breakdown
    instruction_breakdown_str = response.choices[0].message.content.strip()

    # Parse the response into a dictionary
    try:
        parsed = ast.literal_eval(instruction_breakdown_str)
        if isinstance(parsed, dict):
            normalized = {
                "landmarks": parsed.get("landmarks", []),
                "navigation_actions": parsed.get("navigation_actions", []),
                "behavioral_actions": parsed.get("behavioral_actions", []),
                "behavioral_targets": parsed.get("behavioral_targets", [])
            }
            return normalized["landmarks"]
    except Exception as e:
        print(f"Error parsing the instruction breakdown: {e}")
        return []