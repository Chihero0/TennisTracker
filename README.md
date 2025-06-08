# Tennis Tracker - 网球追踪系统
# Tennis Tracker - Tennis Tracking System

## 项目简介 | Project Introduction

这是一个基于YOLOv8的网球追踪系统，能够实时检测和追踪网球比赛中的球员和网球。该系统支持视频处理、球场检测、球轨迹追踪等功能。

This is a tennis tracking system based on YOLOv8 that can detect and track players and tennis balls in real-time. The system supports video processing, court detection, and ball trajectory tracking.

## 功能特点 | Features

- 实时球员和网球检测 | Real-time player and tennis ball detection
- 球场边界识别 | Court boundary detection
- 球轨迹追踪和预测 | Ball trajectory tracking and prediction
- 支持视频处理 | Video processing support
- Core ML模型导出 | Core ML model export

## 环境要求 | Requirements

- Python 3.8+
- CUDA支持（推荐）| CUDA support (recommended)
- 依赖包 | Dependencies:
  ```
  numpy>=1.24.0
  opencv-python>=4.8.0
  pandas==2.2.2
  torch>=2.0.0
  torchvision>=0.21.0
  ultralytics>=8.0.0
  coremltools>=7.0
  ```

## 安装说明 | Installation

1. 克隆仓库 | Clone the repository:
   ```bash
   git clone [repository-url]
   cd TennisTracker
   ```

2. 安装依赖 | Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. 下载预训练模型 | Download pre-trained model:
   ```bash
   python download_model.py
   ```

## 使用方法 | Usage

1. 训练模型 | Train the model:
   ```bash
   python download_model.py
   ```

2. 转换为Core ML格式 | Convert to Core ML format:
   ```bash
   python convert_to_coreml.py
   ```

3. 处理视频 | Process video:
   ```bash
   python tennis_analysis.py
   ```

## 项目结构 | Project Structure

```
TennisTracker/
├── convert_to_coreml.py    # Core ML模型转换脚本
├── download_model.py       # 模型下载和训练脚本
├── tennis_analysis.py      # 主要分析脚本
├── tennis.yaml            # 数据集配置文件
└── requirements.txt       # 项目依赖文件
```

## 注意事项 | Notes

- 确保有足够的GPU内存用于模型训练
- 视频处理可能需要较长时间，取决于视频长度和硬件配置
- 建议使用高质量的视频源以获得最佳效果

## 许可证 | License

[添加许可证信息] | [Add license information]

## 联系方式 | Contact

[添加联系方式] | [Add contact information] 