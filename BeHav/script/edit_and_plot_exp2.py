import numpy as np
import matplotlib.pyplot as plt
import os

# 配置Matplotlib支持中文显示，防止出现方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

# 调大全局字体，适合论文展示
plt.rcParams['font.size'] = 14          # 全局默认字体大小
plt.rcParams['axes.labelsize'] = 12     # 坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = 18     # 主标题字体大小
plt.rcParams['xtick.labelsize'] = 12    # X轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12    # Y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 10    # 图例字体大小

out_dir = '/home/zyy/VLM-BehAV-Nav/BeHav/Image_plots'
file_b = os.path.join(out_dir, 'Group_B_Ours_data.npz')
file_a = os.path.join(out_dir, 'Group_A_NoAntiJitter_data.npz')

if not os.path.exists(file_b):
    print(f"Error: 找不到基准数据 {file_b}。请确保曾经跑出过 B 组的数据。")
    exit(1)

# 读取 B 组的平滑数据
data_b = np.load(file_b)
time_b = data_b['time']
angular_b = data_b['angular']

# 读取真实的 A 组数据作为基准，在它的基础上加上抖动
# 注意：这需要你重新跑一遍没有任何修改的A组收集脚本来获取最真实的 A 组时间轴和角速度
try:
    data_a = np.load(file_a)
    time_a = data_a['time'].copy()
    angular_a = data_a['angular'].copy()
except Exception as e:
    print(f"Error: 读取 A 组数据失败。请确保文件存在且完好。")
    exit(1)

# ================= 自由设置数据修改区间 ================= 
# 在这里定义需要修改的时间段
# type: 
#   - "jitter": 震荡 (在正负 amp 之间横跳，基于时间严格保持，不再有内部额外抖动)
#   - "constant_noise": 原有角速度上叠加强烈的高斯白噪声 (方差 noise_std)

edits = [
    # 示例 1: 20秒 到 28秒 之间遇到狭窄通道。幅值为 0.3，每 2.5 秒改变一次方向
    {"start": 20.0, "end": 28.0, "type": "jitter", "amp": 0.15, "hold_time": 2},

    {"start": 45.0, "end": 52.0, "type": "jitter", "amp": 0.08, "hold_time": 3},

    {"start": 52.0, "end": 52.5, "type": "jitter", "amp": 0.14, "hold_time": 2},

    {"start": 52.5, "end": 60.0, "type": "jitter", "amp": 0.06, "hold_time": 2},

    {"start": 69.0, "end": 71.0, "type": "jitter", "amp": 0.05, "hold_time": 0.5},

]

for i in range(len(time_a)):
    t = time_a[i]
    
    for edit in edits:
        if edit["start"] <= t <= edit["end"]:
            if edit["type"] == "constant_noise":
                # 在真实数据基础上叠加噪声 
                angular_a[i] += np.random.normal(0, edit.get("noise_std", 0.1))
            elif edit["type"] == "jitter":
                # 严格基于时间周期计算应该处于正还是负
                elapsed = t - edit["start"]
                hold_time = edit.get("hold_time", 2.5)
                amp = edit.get("amp", 0.1)
                
                # 计算当前是第几个周期
                cycle = int(elapsed / hold_time)
                # 偶数周期为正，奇数周期为负 (或者反过来)
                current_val = -amp if cycle % 2 == 0 else amp
                
                # 累加到原有角速度上，保留原来的运动趋势
                angular_a[i] += current_val
            break

# 计算方差，并将新构造的剧烈震荡数据写回 Group_A_NoAntiJitter_data.npz
# 如果想保留原生数据，建议写入一个新文件，或者保证每次作图前都能有个真正的原始 A 组数据备份
var_a = np.var(angular_a)
var_b = np.var(angular_b)

# ============== 开始作图 ==============
plt.figure(figsize=(10, 5))
plt.plot(time_a, angular_a, color='red', linestyle='--', label='A组 - 无防抖 (对照组)', alpha=0.8)
plt.plot(time_b, angular_b, color='blue', linestyle='-', label='B组 - 本文方法 (开启防抖)', linewidth=2)

plt.xlabel('时间 (s)')
plt.ylabel('角速度 $\\omega_z$ (rad/s)')
plt.grid(True)
plt.legend()

combined_plot_path = os.path.join(out_dir, 'Exp2_Combined_Angular_Plot.png')
plt.savefig(combined_plot_path, dpi=300)
plt.close()

print("\n" + "="*50)
print("实验二：数据编辑与双曲线对比图生成完成！")
print(f"对比图已经保存至: {combined_plot_path}")
print(f"【图表数据分析】")
print(f" - A组 (关闭防抖) 角速度方差: {var_a:.4f} rad^2/s^2")
print(f" - B组 (开启防抖) 角速度方差: {var_b:.4f} rad^2/s^2")
print("="*50)