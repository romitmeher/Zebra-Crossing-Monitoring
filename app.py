from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
from zebra_crossing_monitor import ZebraCrossingMonitor
import json
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

monitor = None
video_path = None

def generate_frames():
    global monitor, video_path
    if not video_path or not os.path.exists(video_path):
        return
    
    cap = cv2.VideoCapture(video_path)
    monitor = ZebraCrossingMonitor()
    
    monitor.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            processed_frame = monitor.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global video_path
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        video_path = filename
        return jsonify({'message': 'File uploaded successfully', 'filename': file.filename})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats')
def get_stats():
    if monitor:
        return jsonify({
            'total_crossings': monitor.total_crossings,
            'crossing_history': monitor.crossing_history,
            'progress': monitor.progress,
            'zebra_crossing_detected': monitor.zebra_crossing_detected
        })
    return jsonify({
        'total_crossings': 0,
        'crossing_history': [],
        'progress': 0,
        'zebra_crossing_detected': False
    })

@app.route('/download_data')
def download_data():
    if monitor:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            'total_crossings': monitor.total_crossings,
            'crossing_history': monitor.crossing_history,
            'timestamp': timestamp,
            'zebra_crossing_detected': monitor.zebra_crossing_detected
        }
        
        filename = f'crossing_data_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        return jsonify({'filename': filename})
    return jsonify({'error': 'No data available'}), 400

if __name__ == '__main__':
    app.run(debug=True) 