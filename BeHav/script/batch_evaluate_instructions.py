import sys
import os
import ast
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instruction_processor import get_instruction_breakdown, get_ith_key_list, client
import json

def evaluate_instruction(instruction, level_key):
    """
    提交给大模型获取解析字典，并召唤 LLM 充当“AI 裁判”(LLM-as-a-Judge)，智能判定特征提取是否无遗漏且正确。
    """
    print(f"  [测试] {instruction}")
    breakdown = get_instruction_breakdown(instruction)
    
    prompt = f'''
作为智能机器人导航系统的严谨评测员，请你判断“导航指令解析模块”是否准确无误地从用户长句中提取了所有关键特征。

测试场景难度: {level_key}
用户指令: "{instruction}"
被测模块输出: {json.dumps(breakdown, ensure_ascii=False)}

评判准则:
1. landmarks 需包含主要的目的地。
2. behavioral_targets 需包含句子要求的所有需要避开或需要保持在上面的所有障碍物与表面，以及所有穿越的通道（如狭窄通道、以及它两旁的参照物）。如果被测模块输出的 behavioral_targets 列表中切实包含了原句中提到的所有关键障碍物、约束面及其边界实体，即为正确。
3. 请仔细核对被测模块给出的列表元素，不要凭空说它漏抓了某个实体（如果列表中明确存在该实体，请务必判定为存在）。

请你首先简要进行一步步分析，看看原句有哪些约束体，被测输出又有哪些约束体。
最后严格以JSON格式输出最终判定。格式：
```json
{{"pass": true或者是false, "reason": "如果false写明原因，如果true写'解析完美'"}}
```
'''
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_TEXT_MODEL", "qwen-plus"),
            messages=[
                {"role": "system", "content": "You are a strict, objective AI judge for robotic NLP pipelines."},
                {"role": "user", "content": prompt}
            ]
        )
        res_str = response.choices[0].message.content.strip()
        
        # 使用更稳健的方式提取最后的JSON块
        import re
        match = re.search(r'```json\s*(\{.*?\})\s*```', res_str, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # 兼容它没有用markdown块包裹的情况
            match = re.search(r'(\{"pass":.*?\})', res_str, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = res_str

        judgement = json.loads(json_str)
        is_valid = judgement.get("pass", False)
        fail_reason = judgement.get("reason", "大模型裁判未给出具体原因")
        
        if all(not v for v in breakdown.values()):
            is_valid = False
            fail_reason = "由于内部失败，解析模块返回了全空字典"
            
    except Exception as e:
        is_valid = False
        fail_reason = f"AI裁判运行异常: {e}"
        
    return is_valid, breakdown, [fail_reason]


def main():
    txt_path = os.path.join(os.path.dirname(__file__), "input", "test_instructions.txt")
    if not os.path.exists(txt_path):
        print(f"文件不存在: {txt_path}")
        return

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_level = None
    results = {"Level1": {"total": 0, "pass": 0}, 
               "Level2": {"total": 0, "pass": 0}, 
               "Level3": {"total": 0, "pass": 0}}
    
    print("================ 开始自动化批量评测 Qwen 指令解构精度 ================\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("==="):
            if "难度等级 1" in line:
                current_level = "Level1"
            elif "难度等级 2" in line:
                current_level = "Level2"
            elif "难度等级 3" in line:
                current_level = "Level3"
            print(f"\n{line}")
            continue
            
        if current_level and line[0].isdigit():
            # 剥离 "1. " 这类前缀
            instruction = line.split(". ", 1)[1]
            
            results[current_level]["total"] += 1
            is_valid, breakdown, reason = evaluate_instruction(instruction, current_level)
            
            if is_valid:
                results[current_level]["pass"] += 1
            else:
                print(f"    [失败] 原因: {reason}")
                print(f"    [模型输出] {breakdown}\n")

    # 打印最终报告 (这也是你论文可以直接抄的真实数据)
    print("\n================ 评测报告 (可填入论文 6.2 节) ================")
    total_samples = 0
    total_passed = 0
    for level, data in results.items():
        if data["total"] > 0:
            acc = (data["pass"] / data["total"]) * 100
            print(f"{level} - 测试数量: {data['total']}, 成功: {data['pass']}, 准确率: {acc:.2f}%")
            total_samples += data["total"]
            total_passed += data["pass"]
            
    if total_samples > 0:
        overall_acc = (total_passed / total_samples) * 100
        print(f"\n总计 - 测试数量: {total_samples}, 成功: {total_passed}, 平均准确率: {overall_acc:.2f}%")

if __name__ == '__main__':
    main()