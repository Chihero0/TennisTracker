import cv2
import numpy as np
from ultralytics import YOLO
import time

class TennisAnalyzer:
    def __init__(self):
        # 加载预训练的YOLOv8模型
        self.model = YOLO('yolov8n.pt')
        
        # 设置检测参数
        self.conf_threshold = 0.15  # 降低检测阈值以提高检测率
        self.iou_threshold = 0.4    # 调整IOU阈值
        
        # 定义类别ID
        self.PERSON_CLASS = 0
        self.SPORTS_BALL_CLASS = 32
        
        # 初始化球的位置历史
        self.ball_positions = []
        self.max_history = 30  # 保存最近30帧的位置
        
        # 初始化速度预测参数
        self.last_ball_pos = None
        self.ball_velocity = None
        
        # 球场检测参数
        self.min_court_area = 50000  # 最小球场面积
        self.max_court_area = 300000  # 最大球场面积
        self.min_aspect_ratio = 1.5  # 最小长宽比
        self.max_aspect_ratio = 3.0  # 最大长宽比
        
    def detect_court(self, frame):
        """改进的球场检测算法"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用自适应阈值处理
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # 形态学操作
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 找到最大的轮廓
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            
            # 获取最小外接矩形
            rect = cv2.minAreaRect(max_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # 计算矩形的宽和高
            width = rect[1][0]
            height = rect[1][1]
            aspect_ratio = max(width, height) / min(width, height)
            
            # 计算中心点
            center = np.mean(box, axis=0)
            
            # 判断是否为球场
            if (self.min_court_area < area < self.max_court_area and
                self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
                
                # 计算轮廓的复杂度
                perimeter = cv2.arcLength(max_contour, True)
                complexity = perimeter * perimeter / (4 * np.pi * area)
                
                # 如果复杂度在合理范围内，认为是球场
                if 1.0 < complexity < 2.0:
                    return True, box, center
        
        return False, None, None

    def process_video(self, input_path, output_path):
        # 打开视频文件
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {input_path}")
            return
            
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_time = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            
            # 检测球场
            court_detected, court_box, court_center = self.detect_court(frame)
            
            # 使用YOLO进行检测
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)[0]
            
            # 处理检测结果
            for det in results.boxes.data:
                x1, y1, x2, y2, conf, cls = det
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # 根据类别绘制不同颜色的框
                if int(cls) == self.PERSON_CLASS:
                    # 绘制球员检测框（绿色）
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Player {conf:.2f}', (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                elif int(cls) == self.SPORTS_BALL_CLASS:
                    # 只处理小尺寸的检测框（网球通常较小）
                    if (x2 - x1) < 40 and (y2 - y1) < 40:  # 更严格的尺寸限制
                        # 更新球的位置历史
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        self.update_ball_positions(center)
                        
                        # 预测球的位置
                        predicted_pos = self.predict_ball_position()
                        
                        # 绘制球的检测框（红色）
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f'Ball {conf:.2f}', (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # 绘制预测位置（黄色点）
                        if predicted_pos:
                            cv2.circle(frame, predicted_pos, 5, (0, 255, 255), -1)
            
            # 绘制球的轨迹
            self.draw_ball_trajectory(frame)
            
            # 绘制帧计数
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 如果检测到球场，绘制球场边界
            if court_detected:
                cv2.drawContours(frame, [court_box], 0, (255, 255, 255), 2)
                cv2.circle(frame, (int(court_center[0]), int(court_center[1])), 5, (255, 255, 255), -1)
            
            # 计算处理时间
            end_time = time.time()
            process_time = (end_time - start_time) * 1000
            total_time += process_time
            
            # 显示处理时间
            cv2.putText(frame, f'Time: {process_time:.1f}ms', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 写入输出视频
            out.write(frame)
            
            frame_count += 1
            
            # 每100帧显示一次平均处理时间
            if frame_count % 100 == 0:
                avg_time = total_time / frame_count
                print(f"已处理 {frame_count} 帧，平均处理时间: {avg_time:.1f}ms")
        
        # 释放资源
        cap.release()
        out.release()
        print(f"视频处理完成，共处理 {frame_count} 帧")
        print(f"输出视频已保存至: {output_path}")

    def update_ball_positions(self, position):
        """更新球的位置历史"""
        self.ball_positions.append(position)
        if len(self.ball_positions) > self.max_history:
            self.ball_positions.pop(0)
            
        # 更新速度预测
        if len(self.ball_positions) >= 2:
            current_pos = np.array(position)
            if self.last_ball_pos is not None:
                # 计算速度（使用指数平滑）
                new_velocity = current_pos - self.last_ball_pos
                if self.ball_velocity is None:
                    self.ball_velocity = new_velocity
                else:
                    # 使用指数平滑更新速度
                    self.ball_velocity = 0.7 * self.ball_velocity + 0.3 * new_velocity
            self.last_ball_pos = current_pos

    def predict_ball_position(self):
        """预测球的下一个位置"""
        if self.last_ball_pos is not None and self.ball_velocity is not None:
            # 预测未来5帧的位置
            predicted_pos = self.last_ball_pos + self.ball_velocity * 5
            return (int(predicted_pos[0]), int(predicted_pos[1]))
        return None

    def draw_ball_trajectory(self, frame):
        """绘制球的运动轨迹"""
        if len(self.ball_positions) >= 2:
            # 绘制轨迹线（黄色）
            for i in range(len(self.ball_positions) - 1):
                cv2.line(frame, self.ball_positions[i], self.ball_positions[i+1],
                        (0, 255, 255), 2)

if __name__ == "__main__":
    analyzer = TennisAnalyzer()
    analyzer.process_video("input_video.mp4", "output_video.mp4")