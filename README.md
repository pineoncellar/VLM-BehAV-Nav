# VLM-BehAV-Nav
**VLM-BehAV-Nav** 是一款基于视觉语言模型（Visual Language Model, VLM）的具身智能视觉语言导航系统。该系统将自然语言指令与开放词汇（Open-Vocabulary）的视觉语义感知相结合，能够在未知的复杂环境中实现阿克曼（类车）底盘机器人的自主建图、语义寻找与路径规划。

## 🌟 核心特性
- **自然语言指令理解**：解析人类输入的自然语言文本，将其转化为可供机器人执行的阶段性导航和寻找地标任务。
- **先进的零样本语义感知**：结合 ClipSeg 与 FastSAM 等视觉基础模型，利用相机图像实现对环境中物体的零样本分割与地标识别。
- **分层路径规划大脑**：
  - **全局规划 (Far Planner)**: 基于语义与占据栅格地图的远端寻点规划。
  - **局部局部 (Local Planner)**: 基于阿克曼运动学模型的 MPC (模型预测控制) 轨迹跟踪器以及 Rollout 避障采样器，确保机器人在复杂环境中平滑、安全地运动。
- **ROS & Gazebo 仿真支持**：内置完善的仿真系统与自定义 3D 物理环境，搭载 VLP-16 3D激光雷达与相机的阿克曼小车模型，开箱即用。

## 📂 项目架构

系统主要由以下两大核心模块组成：

### 1. `BeHav/` (算法大脑)
项目的核心算法层，采用 Python 编写，主要负责感知、认知流以及规划：
- **`instruction_processor.py`**: 自然语言指令处理与状态机管理。
- **`landmark_vision.py`**: 核心视觉模块，用于提取地标与特征。
- **`planner/`**: 包含全局规划（Far Planner）与基于 MPC 的局部精准控制（Local Planner）。
- **`map/`**: 深度网格生成与 ClipSeg 语义融合模块，实时构建语义地图。

### 2. `robot_yang/` (仿真环境与机器人平台)
基于 ROS/ROS 2 和 Gazebo 构建的高逼真仿真测试台：
- **车型机器人 (`carlike_robot_description`)**: 通过 xacro 定义的车辆模型，配备深度相机与多线激光雷达。
- **多元测试场景 (`worlds/`)**: 提供了诸如街区（Street）、室外花园（Outdoor Garden）等多种仿真环境用于导航能力验证。

## 🚀 快速开始

本项目依赖 ROS 环境与相应的 Python 算法依赖。启动系统通常分为两步：

1. **启动仿真环境**  
   进入 `robot_yang/` 目录下，启动 ROS Gazebo 环境与机器人节点：
   ```bash
   cd robot_yang
   ./start.sh
   ```

2. **启动视觉与导航算法控制**  
   进入 `BeHav/` 目录下，启动语义视觉与规划节点：
   ```bash
   cd BeHav
   ./run_ros_planning.sh
   ```

      ```bash
   cd BeHav
   ./run.sh
   ```
