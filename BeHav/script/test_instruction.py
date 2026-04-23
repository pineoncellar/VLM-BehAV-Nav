import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instruction_processor import get_instruction_breakdown, get_similarity_scores, calculate_input_action_costs, get_ith_key_list

def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger("test_instruction")

    print("\n" + "="*50)
    print("    [测试模块] 指令解析 (Instruction Processing)")
    print("="*50)
    instruction = input(">>> 请输入您的自然语言导航指令: ").strip()
    if not instruction:
        instruction = "Walk to the table and avoid the person"
        print(f"使用默认指令: {instruction}")

    logger.info(f"Starting parsing for: {instruction}")

    # 1. 启动并测试大语言模型指令分解
    breakdown = get_instruction_breakdown(instruction)

    landmarks = get_ith_key_list(breakdown, 1)
    nav_acts = get_ith_key_list(breakdown, 2)
    beh_acts = get_ith_key_list(breakdown, 3)
    beh_targets = get_ith_key_list(breakdown, 4)

    logger.info(f"Landmarks: {landmarks}")
    logger.info(f"Navigation Actions: {nav_acts}")
    logger.info(f"Behavioral Actions: {beh_acts}")
    logger.info(f"Behavioral Targets: {beh_targets}")

    reference_list = ['Stay on', 'Avoid', 'Yield', 'Stop']
    reference_costs = [0, 0.5, 0.7, 1.0]

    # 2. 启动并测试代价计算模型
    if beh_acts is not None and len(beh_acts) > 0:
        scores = get_similarity_scores(beh_acts, reference_list)
        costs = calculate_input_action_costs(scores, reference_costs)
        logger.info(f"Calculated Behavioral Costs: {costs}")
    else:
        logger.info("No Behavioral Actions extracted, skip cost calculation.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试结束")
