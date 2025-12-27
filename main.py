import cv2
import numpy as np
from datetime import datetime
import os

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Optimized color ranges for better detection
color_ranges = {
    'red': {'lower1': [0, 100, 100], 'upper1': [10, 255, 255], 
            'lower2': [170, 100, 100], 'upper2': [179, 255, 255]},
    'green': {'lower1': [40, 100, 100], 'upper1': [80, 255, 255]},
    'blue': {'lower1': [100, 100, 100], 'upper1': [130, 255, 255]},
    'yellow': {'lower1': [20, 100, 100], 'upper1': [30, 255, 255]}
}

current_color = 'red'

def create_optimized_mask(hsv, color):
    """Create optimized mask for specific color"""
    ranges = color_ranges[color]
    
    # Create mask for primary range
    mask = cv2.inRange(hsv, np.array(ranges['lower1']), np.array(ranges['upper1']))
    
    # Add secondary range for red (wraps around HSV)
    if 'lower2' in ranges:
        mask2 = cv2.inRange(hsv, np.array(ranges['lower2']), np.array(ranges['upper2']))
        mask = cv2.bitwise_or(mask, mask2)
    
    return mask

def enhance_mask(mask):
    """Enhanced mask processing for cleaner detection"""
    # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    # Fill gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    # Smooth edges
    mask = cv2.medianBlur(mask, 15)
    # Expand detected area slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask

cap = cv2.VideoCapture(0)
cv2.namedWindow("bars")

# Create trackbars for fine-tuning (optional)
cv2.createTrackbar("Sensitivity", "bars", 50, 100, lambda x: None)

# Add color selection trackbar
cv2.createTrackbar("Color", "bars", 0, 3, lambda x: None)  # 0=red, 1=green, 2=blue, 3=yellow
color_names = ['red', 'green', 'blue', 'yellow']

print("Controls:")
print("- Use 'Color' trackbar: 0=Red, 1=Green, 2=Blue, 3=Yellow")
print("- Use 'Sensitivity' trackbar to fine-tune detection")
print("- Press 'q' to quit")

# Capture background
print("Move away from camera and press SPACE to capture background")
while True:
    ret, background = cap.read()
    cv2.imshow("Background Capture", background)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
cv2.destroyWindow("Background Capture")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get current color selection
    color_index = cv2.getTrackbarPos("Color", "bars")
    current_color = color_names[color_index]
    
    # Get sensitivity for fine-tuning
    sensitivity = cv2.getTrackbarPos("Sensitivity", "bars")
    
    # Create optimized mask for current color
    mask = create_optimized_mask(hsv, current_color)
    
    # Apply sensitivity adjustment
    if sensitivity != 50:  # 50 is default
        adjustment = (sensitivity - 50) * 2  # Scale adjustment
        ranges = color_ranges[current_color]
        
        # Adjust saturation threshold
        lower_sat = max(0, ranges['lower1'][1] - adjustment)
        mask_adjusted = cv2.inRange(hsv, 
                                  np.array([ranges['lower1'][0], lower_sat, ranges['lower1'][2]]),
                                  np.array(ranges['upper1']))
        
        if 'lower2' in ranges:
            mask2_adjusted = cv2.inRange(hsv,
                                       np.array([ranges['lower2'][0], lower_sat, ranges['lower2'][2]]),
                                       np.array(ranges['upper2']))
            mask_adjusted = cv2.bitwise_or(mask_adjusted, mask2_adjusted)
        
        mask = mask_adjusted
    
    # Enhanced mask processing
    mask = enhance_mask(mask)
    
    # Enhanced face and skin protection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))
    
    # Create comprehensive human protection mask
    human_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    # Protect faces with larger area
    for (x, y, w, h) in faces:
        # Expand face area for better protection
        padding = 20
        x1, y1 = max(0, x-padding), max(0, y-padding)
        x2, y2 = min(frame.shape[1], x+w+padding), min(frame.shape[0], y+h+padding)
        cv2.rectangle(human_mask, (x1, y1), (x2, y2), 255, -1)
    
    # Additional skin tone protection
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin_lower = np.array([0, 133, 77], dtype=np.uint8)
    skin_upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, skin_lower, skin_upper)
    
    # Clean skin mask
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    
    # Combine face and skin protection
    human_mask = cv2.bitwise_or(human_mask, skin_mask)
    
    # Remove all human areas from cloth mask
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(human_mask))
    
    # Show current color and mask
    cv2.putText(mask, f'Color: {current_color.upper()}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.imshow("Mask", mask)
    
    # Create invisibility effect
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(frame, frame, mask=mask_inv)
    background_part = cv2.bitwise_and(background, background, mask=mask)
    final = cv2.add(result, background_part)
    
    cv2.imshow("Invisibility Cloak", final)
    cv2.imshow("Original", frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()