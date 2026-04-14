import cv2
import pyttsx3
import threading
import os
from collections import Counter
from ultralytics import YOLO

# ==========================================
# 1. THREAD-SAFE VOICE LOGIC
# ==========================================
def speak(text):
    def run_speech():
        try:
            # Local engine instance to prevent 'run loop' errors
            local_engine = pyttsx3.init()
            local_engine.setProperty('rate', 160)
            local_engine.say(text)
            local_engine.runAndWait()
            del local_engine
        except Exception as e:
            print(f"Speech error: {e}")
            
    threading.Thread(target=run_speech, daemon=True).start()

# ==========================================
# 2. MODEL & BUFFER SETUP
# ==========================================
model = YOLO("best.pt")

detection_buffer = []
BUFFER_SIZE = 12 # Slightly smaller for faster response

cap = cv2.VideoCapture(0)
last_spoken_gesture = None

print("--- System Active: Live Confidence + Smoothing ---")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    box_size = 320
    x1, y1 = (w - box_size) // 2, (h - box_size) // 2
    x2, y2 = x1 + box_size, y1 + box_size
    roi = frame[y1:y2, x1:x2]

    # Predict
    results = model.predict(roi, conf=0.4, imgsz=320, verbose=False)
    
    current_frame_gesture = None
    current_conf = 0.0

    if len(results[0].boxes) > 0:
        top_box = results[0].boxes[0]
        cls_id = int(top_box.cls[0])
        current_conf = float(top_box.conf[0])
        current_frame_gesture = model.names[cls_id]

    # --- THE SMOOTHING ENGINE ---
    # Add to buffer
    if current_frame_gesture:
        detection_buffer.append(current_frame_gesture)
    else:
        detection_buffer.append("None")

    if len(detection_buffer) > BUFFER_SIZE:
        detection_buffer.pop(0)

    # Calculate most frequent
    data = Counter(detection_buffer)
    most_common_gesture, count = data.most_common(1)[0]

    # --- UI & LOGIC ---
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # 1. SHOW LIVE DATA (What the AI sees right this millisecond)
    if current_frame_gesture:
        cv2.putText(frame, f"Live: {current_frame_gesture} ({current_conf:.2f})", 
                    (x1, y1 + box_size + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # 2. SHOW STABLE DATA (What the system trusts)
    if most_common_gesture != "None" and count > (BUFFER_SIZE * 0.6):
        # UI Feedback
        display_text = f"CONFIRMED: {most_common_gesture}"
        cv2.putText(frame, display_text, (x1, y1 - 15), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Voice Trigger
        if most_common_gesture != last_spoken_gesture:
            speak(most_common_gesture)
            last_spoken_gesture = most_common_gesture
    else:
        if most_common_gesture == "None":
            last_spoken_gesture = None
        cv2.putText(frame, "Waiting for stable signal...", (x1, y1 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imshow("Hand-to-Voice Full Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()