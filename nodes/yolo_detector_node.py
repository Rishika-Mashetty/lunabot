import cv2
from ultralytics import YOLO
from gtts import gTTS
import playsound
import os
import time

# 🔗 Your IP camera or Flask video stream URL
url = "http://192.168.1.4:5000/video_feed"

# Load YOLO model
yolo = YOLO("yolov8m.pt")

# Function for speech output
def speak(text):
    print(f"🔊 {text}")
    tts = gTTS(text=text, lang='en')
    tts.save("speech.mp3")
    playsound.playsound("speech.mp3", True)
    os.remove("speech.mp3")

# Connect to the live stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ Could not open video stream. Check URL or network.")
    exit()

last_alert_time = 0
alert_interval = 10  # seconds between repeated alerts

print("🚀 Live hazard detection started...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Stream ended or disconnected. Reconnecting...")
        time.sleep(2)
        cap = cv2.VideoCapture(url)
        continue

    # Run YOLO detection
    results = yolo(frame)

    # Check for detected objects
    if len(results[0].boxes) > 0:
        current_time = time.time()
        if current_time - last_alert_time > alert_interval:
            count = len(results[0].boxes)
            speak(f"Hazard detected: {count} object{'s' if count > 1 else ''}")
            last_alert_time = current_time

    # Optional: Display the live video feed with bounding boxes
    annotated = results[0].plot()
    cv2.imshow("LunaBot Hazard Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
