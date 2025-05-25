import cv2
import numpy as np
from ultralytics import YOLO
import time
import json
from datetime import datetime
import os
from collections import deque

class ZebraCrossingMonitor:
    def __init__(self):
        self.model = YOLO('yolov8x.pt')
        self.roi_points = None
        self.crossing_line = None
        self.total_crossings = 0
        self.crossing_history = []
        self.tracked_pedestrians = set()
        self.pedestrian_positions = {}
        self.pedestrian_directions = {}
        self.last_detection_time = {}
        self.frame_count = 0
        self.process_every_n_frames = 1
        self.last_processed_frame = None
        self.processing_time = 0
        self.progress = 0
        self.total_frames = 0
        self.current_frame = 0
        self.zebra_crossing_detected = False
        self.zebra_detection_history = deque(maxlen=30)
        self.colors = {
            'roi': (0, 255, 0),
            'crossing_line': (0, 0, 255),
            'pedestrian': (255, 0, 0),
            'crossing': (0, 255, 255),
            'zebra': (255, 255, 0)
        }
        
    def set_roi(self, frame):
        height, width = frame.shape[:2]
        self.roi_points = np.array([
            [width * 0.1, height * 0.2],
            [width * 0.9, height * 0.2],
            [width * 0.9, height * 0.9],
            [width * 0.1, height * 0.9]
        ], np.int32)
        self.crossing_line = int(height * 0.6)
        
    def detect_pedestrians(self, frame):
        scale_percent = 75
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        resized_frame = cv2.resize(frame, (width, height))
        
        results = self.model(resized_frame, 
                           classes=0,
                           conf=0.45,
                           iou=0.5,
                           verbose=False)
        
        boxes = results[0].boxes.data.cpu().numpy()
        if len(boxes) > 0:
            boxes[:, :4] *= (100 / scale_percent)
        return boxes
        
    def is_in_zebra_crossing(self, bbox):
        x1, y1, x2, y2 = bbox[:4]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_points], 255)
        
        feet_y = y2
        return (mask[int(center_y), int(center_x)] == 255 and 
                mask[int(feet_y), int(center_x)] == 255)
        
    def calculate_direction(self, positions):
        if len(positions) < 3:
            return None
            
        x = np.arange(len(positions))
        y = np.array(positions)
        slope = np.polyfit(x, y, 1)[0]
        
        if abs(slope) < 0.5:
            return None
            
        return 'up' if slope < 0 else 'down'
        
    def is_crossing(self, bbox, ped_id):
        if not self.is_in_zebra_crossing(bbox):
            return False
            
        x1, y1, x2, y2 = bbox[:4]
        center_y = (y1 + y2) / 2
        buffer = 30
        
        if ped_id not in self.pedestrian_positions:
            self.pedestrian_positions[ped_id] = []
            self.pedestrian_directions[ped_id] = None
            
        self.pedestrian_positions[ped_id].append(center_y)
        
        if len(self.pedestrian_positions[ped_id]) > 20:
            self.pedestrian_positions[ped_id] = self.pedestrian_positions[ped_id][-20:]
            
        self.pedestrian_directions[ped_id] = self.calculate_direction(self.pedestrian_positions[ped_id])
        
        if len(self.pedestrian_positions[ped_id]) >= 5:
            prev_y = self.pedestrian_positions[ped_id][-5]
            curr_y = self.pedestrian_positions[ped_id][-1]
            
            if self.pedestrian_directions[ped_id] == 'up':
                if prev_y > (self.crossing_line + buffer + 10) and curr_y < (self.crossing_line - buffer - 10):
                    return True
            elif self.pedestrian_directions[ped_id] == 'down':
                if prev_y < (self.crossing_line - buffer - 10) and curr_y > (self.crossing_line + buffer + 10):
                    return True
        return False
    
    def draw_visualization(self, frame, ped, ped_id, is_crossing=False):
        x1, y1, x2, y2 = map(int, ped[:4])
        color = self.colors['crossing'] if is_crossing else self.colors['pedestrian']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if ped_id in self.pedestrian_directions and self.pedestrian_directions[ped_id]:
            direction = self.pedestrian_directions[ped_id]
            arrow_start = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            arrow_end = (arrow_start[0], arrow_start[1] - 30 if direction == 'up' else arrow_start[1] + 30)
            cv2.arrowedLine(frame, arrow_start, arrow_end, color, 2)
        text_y = y1 - 10
        if is_crossing:
            cv2.putText(frame, 'Crossing', (x1, text_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors['crossing'], 2)
            text_y -= 30
        cv2.putText(frame, f'Conf: {ped[4]:.2f}', (x1, text_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def detect_zebra_crossing(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        edges = cv2.Canny(thresh, 30, 150)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        lines = cv2.HoughLinesP(dilated, 1, np.pi/180,
                               threshold=80,
                               minLineLength=80,
                               maxLineGap=20)
        
        if lines is not None:
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                
                if angle < 30 or angle > 150:
                    horizontal_lines.append(line[0])
            
            if len(horizontal_lines) >= 3:
                horizontal_lines.sort(key=lambda x: (x[1] + x[3])/2)
                
                spacings = []
                for i in range(len(horizontal_lines)-1):
                    y1 = (horizontal_lines[i][1] + horizontal_lines[i][3])/2
                    y2 = (horizontal_lines[i+1][1] + horizontal_lines[i+1][3])/2
                    spacings.append(abs(y2 - y1))
                
                if len(spacings) >= 2:
                    mean_spacing = np.mean(spacings)
                    std_spacing = np.std(spacings)
                    
                    if std_spacing < mean_spacing * 0.3:
                        self.zebra_detection_history.append(True)
                        if sum(self.zebra_detection_history) >= len(self.zebra_detection_history) * 0.7:
                            self.zebra_crossing_detected = True
                            for line in horizontal_lines:
                                cv2.line(frame, (line[0], line[1]), (line[2], line[3]), 
                                       self.colors['zebra'], 2)
                            return True
        
        self.zebra_detection_history.append(False)
        if sum(self.zebra_detection_history) < len(self.zebra_detection_history) * 0.7:
            self.zebra_crossing_detected = False
        return False

    def process_frame(self, frame):
        start_time = time.time()
        if self.roi_points is None:
            self.set_roi(frame)
            self.frame_height, self.frame_width = frame.shape[:2]
        
        self.current_frame += 1
        self.progress = (self.current_frame / self.total_frames) * 100 if self.total_frames > 0 else 0
        
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            if self.last_processed_frame is not None:
                cv2.putText(frame, f'Processing Time: {self.processing_time:.2f}ms', (10, 110),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['roi'], 2)
                return frame
            self.last_processed_frame = frame.copy()

        zebra_detected = self.detect_zebra_crossing(frame)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_points], 255)
        cv2.polylines(frame, [self.roi_points], True, self.colors['roi'], 2)
        cv2.line(frame, (0, self.crossing_line), 
                (frame.shape[1], self.crossing_line), self.colors['crossing_line'], 2)
        
        pedestrians = self.detect_pedestrians(frame)
        current_pedestrians = set()
        current_time = time.time()
        
        for ped in pedestrians:
            x1, y1, x2, y2, conf, cls = ped
            if conf > 0.45:
                ped_id = f"{int(x1)}_{int(y1)}_{int(x2-x1)}_{int(y2-y1)}"
                current_pedestrians.add(ped_id)
                self.last_detection_time[ped_id] = current_time
                
                if self.is_in_zebra_crossing(ped):
                    is_crossing = self.is_crossing(ped, ped_id) and ped_id not in self.tracked_pedestrians
                    self.draw_visualization(frame, ped, ped_id, is_crossing)
                    
                    if is_crossing:
                        self.total_crossings += 1
                        self.crossing_history.append({
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'position': (int(x1), int(y1), int(x2), int(y2)),
                            'direction': self.pedestrian_directions[ped_id],
                            'confidence': float(conf)
                        })
                        self.tracked_pedestrians.add(ped_id)
        
        self.tracked_pedestrians = self.tracked_pedestrians.intersection(current_pedestrians)
        for ped_id in list(self.pedestrian_positions.keys()):
            if ped_id not in current_pedestrians:
                if current_time - self.last_detection_time.get(ped_id, 0) > 2.0:
                    del self.pedestrian_positions[ped_id]
                    del self.pedestrian_directions[ped_id]
                    if ped_id in self.last_detection_time:
                        del self.last_detection_time[ped_id]
        
        cv2.putText(frame, f'Total Crossings: {self.total_crossings}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['roi'], 2)
        cv2.putText(frame, f'Current Pedestrians: {len(current_pedestrians)}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['roi'], 2)
        cv2.putText(frame, f'Zebra Crossing: {"Detected" if zebra_detected else "Not Detected"}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['zebra'], 2)
        
        self.processing_time = (time.time() - start_time) * 1000
        cv2.putText(frame, f'Processing Time: {self.processing_time:.2f}ms', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['roi'], 2)
        
        self.last_processed_frame = frame.copy()
        return frame
    
    def save_crossing_data(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            'total_crossings': self.total_crossings,
            'crossing_history': self.crossing_history,
            'timestamp': timestamp,
            'settings': {
                'model': 'yolov8x',
                'confidence_threshold': 0.45,
                'roi_size': {
                    'width': [0.1, 0.9],
                    'height': [0.2, 0.9]
                }
            }
        }
        filename = f'crossing_data_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Crossing data saved to {filename}") 