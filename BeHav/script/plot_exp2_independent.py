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
file_a = os.path.join(out_dir, 'Group_A_NoAntiJitter_data.npz')
file_b = os.path.join(out_dir, 'Group_B_Ours_data.npz')

if not os.path.exists(file_a) or not os.path.exists(file_b):
    print("找不到对应的数据文件，请确保目录 Image_plots 下拥有这两份 npz 文件。")
    exit(1)

# 读取两组数据
data_a = np.load(file_a)
time_a = data_a['time']
angular_a = data_a['angular']

data_b = np.load(file_b)
time_b = data_b['time']
angular_b = data_b['angular']

# 创建具有上下两个子图的画布(方便横向时间与纵向幅度独立对比)
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharey=True)

# 绘制A组(无防抖)
axes[0].plot(time_a, angular_a, color='red', linestyle='--', label='A组 - 无防抖', linewidth=1.5)
axes[0].set_title('A组: 无防抖 (对照组)')
axes[0].set_ylabel('角速度 $\\omega_z$ (rad/s)')
axes[0].grid(True)
axes[0].legend()

# 绘制B组(有防抖)
axes[1].plot(time_b, angular_b, color='blue', linestyle='-', label='B组 - 本文方法', linewidth=1.5)
axes[1].set_title('B组: 本文方法 (开启防抖)')
axes[1].set_xlabel('时间 (s)')
axes[1].set_ylabel('角速度 $\\omega_z$ (rad/s)')
axes[1].grid(True)
axes[1].legend()

# 自动调整布局防遮挡
plt.tight_layout()

out_file = os.path.join(out_dir, 'Exp2_Individual_Plots_Subplots.png')
plt.savefig(out_file, dpi=300)
plt.close()

print(f"分别绘制的子图已保存至: {out_file}")
