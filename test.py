import cv2
import os

video_path = r"C:\Users\rishi\OneDrive\Desktop\learnings\projects\lunabot\data\room_graph\WhatsApp Video 2025-06-15 at 18.15.58_1b92e557.mp4"
output_img_path = r"C:\Users\rishi\OneDrive\Desktop\learnings\projects\lunabot\data\ending.jpg"

# Check if ending.jpg exists
if not os.path.exists(output_img_path):
    print("⚙️ ending.jpg not found — extracting from video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video file. Check path.")
    else:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 5)  # last few frames
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(output_img_path, frame)
            print(f"✅ Saved ending frame as: {output_img_path}")
        else:
            print("⚠️ Could not extract frame from video.")
        cap.release()
else:
    print("✅ ending.jpg already exists.")

# Verify readability
img = cv2.imread(output_img_path)
print("Image readable:", img is not None)
