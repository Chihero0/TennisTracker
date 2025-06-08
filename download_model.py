import os
import torch
from ultralytics import YOLO

def download_tennis_model():
    print("开始下载网球检测模型...")
    
    # 下载预训练模型
    model = YOLO('yolov8n.pt')
    
    # 设置训练参数
    model.train(
        data='tennis.yaml',  # 数据集配置文件
        epochs=50,           # 训练轮数
        imgsz=640,          # 图像大小
        batch=16,           # 批次大小
        name='tennis_model' # 实验名称
    )
    
    # 将训练好的模型复制到项目目录
    os.rename('runs/detect/tennis_model/weights/best.pt', 'yolov8n-tennis.pt')
    
    print("模型下载和训练完成！")

if __name__ == '__main__':
    download_tennis_model() 