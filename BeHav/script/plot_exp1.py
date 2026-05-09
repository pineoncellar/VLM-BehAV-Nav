import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import re

# 配置Matplotlib支持中文显示，防止出现方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

# 调大全局字体，适合论文展示
plt.rcParams['font.size'] = 14          # 全局默认字体大小
plt.rcParams['axes.labelsize'] = 12     # 坐标轴标签字体大小 (时间(s), 控制频率)
plt.rcParams['axes.titlesize'] = 18     # 主标题字体大小
plt.rcParams['xtick.labelsize'] = 12    # X轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12    # Y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 10    # 图例字体大小

def plot_experiment_1(log_file, vlm_log_file, output_path, start_sec=0.0):
    print(f"Reading data from {log_file}...")
    df = pd.read_csv(log_file)
    
    # 过滤掉不合理的异常高频（比如同一时间发出的多条）以保持图表美观
    df = df[df['hz'] < 100]
    
    # 获取相对时间戳 (从 0 秒开始)
    start_time = df['timestamp'].min()
    df['relative_time'] = df['timestamp'] - start_time
    
    # 根据用户指定的开始秒数进行截断，将这之前的丢弃，并把起点重置为 0
    df = df[df['relative_time'] >= start_sec].copy()
    df['relative_time'] = df['relative_time'] - start_sec
    
    # 使用滚动平均平滑底盘控制频率以更清晰地展示趋势，窗口设为5
    df['hz_smooth'] = df['hz'].rolling(window=5, min_periods=1).mean()
    
    # 读取真实的 VLM 延迟日志
    vision_times = []
    vision_latencies = []
    
    if os.path.exists(vlm_log_file):
        print(f"Reading actual VLM latency data from {vlm_log_file}...")
        with open(vlm_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if "[EXP1_LOG] 视觉管线处理端到端耗时" in line:
                    # 格式示例: [EXP1_LOG] 视觉管线处理端到端耗时: 3.4005 秒, timestamp: 1778345912.496595
                    match = re.search(r'耗时:\s*([\d\.]+)\s*秒.*?timestamp:\s*([\d\.]+)', line)
                    if match:
                        latency = float(match.group(1))
                        timestamp = float(match.group(2))
                        # 加上相对时间
                        rel_time = timestamp - start_time
                        
                        # 应用开始时间截断
                        if rel_time < start_sec:
                            continue
                        rel_time = rel_time - start_sec

                        # 只保留和底盘记录时间段内的数据
                        if -5 <= rel_time <= df['relative_time'].max() + 5:
                            vision_times.append(rel_time)
                            vision_latencies.append(latency)
    else:
        print(f"Warning: Could not find VLM log file at {vlm_log_file}. Please check the file path.")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # 画底盘控制频率 (主坐标轴)
    color = 'tab:blue'
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('控制频率 (Hz)', color=color)
    ax1.plot(df['relative_time'], df['hz_smooth'], color=color, linewidth=2, label='底盘控制频率 (Hz)')
    
    # 增加一个10Hz的基准参考线
    ax1.axhline(y=10, color='blue', linestyle='--', alpha=0.5, label='10 Hz 目标基准')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, max(df['hz_smooth'].max() + 5, 15))
    
    # 画感知端延迟 (副坐标轴)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('VLM 端到端感知延迟 (s)', color=color2)
    
    if vision_times:
        # 用带数据点的折线图表示VLM感知耗时的变化
        ax2.plot(vision_times, vision_latencies, color='tab:red', marker='o', linewidth=2, linestyle='-', label='VLM 感知耗时')
        max_latency = max(vision_latencies)
        ax2.set_ylim(0, max(max_latency + 1, 5)) 
    else:
        ax2.set_ylim(0, 5)

    ax2.tick_params(axis='y', labelcolor=color2)
    
    fig.tight_layout()
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='script/exp1_hz_log.txt')
    parser.add_argument('--input_vlm', default='script/exp1_vlm_log.log')
    parser.add_argument('--output', default='Image_plots/Exp1_Latency_Freq.png')
    parser.add_argument('--start_sec', type=float, default=0.0, help='从几秒开始截断数据（过滤起步时的毛刺）')
    args = parser.parse_args()
    
    plot_experiment_1(args.input, args.input_vlm, args.output, args.start_sec)
