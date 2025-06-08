from ultralytics import YOLO

def convert_yolo_to_coreml():
    # 加载YOLOv8模型
    model = YOLO('yolov8n.pt')
    
    # 直接导出为Core ML格式
    model.export(format='coreml')
    print("模型已成功转换为Core ML格式")

if __name__ == "__main__":
    convert_yolo_to_coreml() 