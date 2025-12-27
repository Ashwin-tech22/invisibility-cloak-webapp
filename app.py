from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import base64
import json
import time
from datetime import datetime
from threading import Lock

app = Flask(__name__)

# Global variables
camera = cv2.VideoCapture(0)
background = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
current_color = 'red'
hsv_values = {'lower_sat': 50, 'lower_val': 50, 'upper_sat': 255, 'upper_val': 255}
show_mask = False
lock = Lock()

# Game variables
game_active = False
game_score = 0
game_lives = 1
last_detection_time = 0
game_start_time = 0
next_check_time = 0
check_interval = 5
check_start_time = 0
checking_duration = 2
import random

# Color ranges
color_ranges = {
    'red': [(0, 10), (170, 179)],
    'green': [(40, 80)],
    'blue': [(100, 130)],
    'yellow': [(20, 30)]
}

def process_frame():
    """Process frame for invisibility effect"""
    global background, current_color, hsv_values, show_mask
    
    ret, frame = camera.read()
    if not ret or background is None:
        return frame, None, 0, 0
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for current color
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for hue_range in color_ranges[current_color]:
        lower_bound = np.array([hue_range[0], hsv_values['lower_sat'], hsv_values['lower_val']])
        upper_bound = np.array([hue_range[1], hsv_values['upper_sat'], hsv_values['upper_val']])
        temp_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.bitwise_or(mask, temp_mask)
    
    # Clean mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    mask = cv2.medianBlur(mask, 19)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Face detection and protection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    face_mask = np.zeros(mask.shape, dtype=np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(face_mask, (x, y), (x+w, y+h), 255, -1)
    
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(face_mask))
    
    # Calculate coverage
    coverage = (np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])) * 100
    
    # Create invisibility effect
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(frame, frame, mask=mask_inv)
    background_part = cv2.bitwise_and(background, background, mask=mask)
    final = cv2.add(result, background_part)
    
    return final, mask if show_mask else None, coverage, len(faces)

def generate_frames():
    """Generate frames for video streaming"""
    while True:
        with lock:
            final, mask, coverage, face_count = process_frame()
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', final)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_background', methods=['POST'])
def capture_background():
    global background
    ret, background = camera.read()
    if ret:
        return jsonify({'status': 'success', 'message': 'Background captured!'})
    return jsonify({'status': 'error', 'message': 'Failed to capture background'})

@app.route('/set_color', methods=['POST'])
def set_color():
    global current_color
    data = request.get_json()
    current_color = data.get('color', 'red')
    return jsonify({'status': 'success', 'color': current_color})

@app.route('/set_hsv', methods=['POST'])
def set_hsv():
    global hsv_values
    data = request.get_json()
    hsv_values.update(data)
    return jsonify({'status': 'success', 'values': hsv_values})

@app.route('/toggle_mask', methods=['POST'])
def toggle_mask():
    global show_mask
    show_mask = not show_mask
    return jsonify({'status': 'success', 'show_mask': show_mask})
@app.route('/get_stats')
def get_stats():
    global game_active, game_lives, next_check_time, check_interval, check_start_time, checking_duration
    
    with lock:
        _, _, coverage, face_count = process_frame()
    
    is_hidden = coverage > 15 and face_count > 0
    cartoon_checking = False
    game_over_reason = ""
    
    # Game logic with continuous countdown timer
    if game_active:
        current_time = time.time()
        
        # Check if countdown reached zero (time to check)
        if current_time >= next_check_time:
            cartoon_checking = True
            
            # Cartoon checks if player is hidden
            if not is_hidden and face_count > 0:
                # Player caught - game over
                game_active = False
                game_lives = 0
                game_over_reason = "YOU ARE CAUGHT!"
            else:
                # Player survived - reset timer with new random countdown
                new_countdown = random.randint(5, 15)  # Random 5-15 seconds
                next_check_time = current_time + new_countdown
                global game_score
                game_score += 10  # Bonus for surviving a check
    
    return jsonify({
        'coverage': round(coverage, 1),
        'faces': face_count,
        'color': current_color,
        'show_mask': show_mask,
        'is_hidden': is_hidden,
        'cartoon_checking': cartoon_checking,
        'game_over_reason': game_over_reason
    })

@app.route('/start_game', methods=['POST'])
def start_game():
    global game_active, game_score, game_lives, game_start_time, last_detection_time, next_check_time
    game_active = True
    game_score = 0
    game_lives = 1
    game_start_time = time.time()
    last_detection_time = 0
    # Start with random countdown (5-15 seconds)
    initial_countdown = random.randint(5, 15)
    next_check_time = time.time() + initial_countdown
    return jsonify({'status': 'success', 'message': f'Game started! First check in {initial_countdown} seconds'})

@app.route('/stop_game', methods=['POST'])
def stop_game():
    global game_active
    game_active = False
    return jsonify({'status': 'success', 'message': 'Game stopped!'})

@app.route('/get_game_stats')
def get_game_stats():
    global game_active, game_score, game_lives, game_start_time, next_check_time
    
    # Calculate time until next cartoon check
    time_to_check = max(0, int(next_check_time - time.time())) if game_active else 0
    
    return jsonify({
        'active': game_active,
        'score': game_score,
        'lives': game_lives,
        'time': int(time.time() - game_start_time) if game_active and game_start_time > 0 else 0,
        'next_check': time_to_check
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)