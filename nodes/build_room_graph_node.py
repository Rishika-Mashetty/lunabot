#3 ending and starting images are saved per room

import os, json, cv2, numpy as np, torch
from PIL import Image
from gtts import gTTS
from IPython.display import Audio, display
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from collections import deque

# --- 📁 Config ---
VIDEO_PATH = "C:/Users/rishi/OneDrive/Desktop/learnings/projects/lunabot/data/room_graph/WhatsApp Video 2025-06-15 at 18.15.58_1b92e557.mp4"
END_IMG_PATH = "C:/Users/rishi/OneDrive/Desktop/learnings/projects/lunabot/data/ending.jpg"
OUTPUT_DIR = "C:/Users/rishi/OneDrive/Desktop/learnings/projects/lunabot/data/room_graph"
FRAME_INTERVAL = 10  # Save every 1 sec if video is 30fps
SIM_THRESHOLD = 0.87
TURN_THRESHOLD_DEGREES = 60
TURN_FLOW_MAG = 2.5
BLUR_THRESHOLD = 100.0
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 🔌 Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
yolo = YOLO("yolov8m.pt")

# --- 🔊 Speak
def speak(text):
    print("🔊", text)
    tts = gTTS(text=text, lang='en')
    tts.save("speech.mp3")
    display(Audio("speech.mp3", autoplay=True))

# --- 🔍 Embedding + Similarity
def get_clip_embedding(img):
    inputs = clip_processor(images=Image.fromarray(img), return_tensors="pt").to(device)
    with torch.no_grad():
        return clip_model.get_image_features(**inputs)[0].cpu().numpy()

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- 📐 Blur Check
def is_clear_image(img, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= threshold

# --- 🧭 Estimate Turn
def estimate_rotation(prev, curr):
    prev_gray, curr_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dx, dy = np.mean(flow[..., 0]), np.mean(flow[..., 1])
    angle = np.degrees(np.arctan2(dy, dx))
    mag = np.sqrt(dx**2 + dy**2)
    if mag < TURN_FLOW_MAG: return "forward", 0
    if angle < -TURN_THRESHOLD_DEGREES: return "left", abs(angle)
    if angle > TURN_THRESHOLD_DEGREES: return "right", abs(angle)
    return "forward", abs(angle)

# --- 🧱 Room Graph Builder
def build_room_graph(video_path):
    cap = cv2.VideoCapture(video_path)
    room_id = 0
    prev_emb, prev_frame = None, None
    frame_idx = 0
    current_room = f"room_{room_id}"
    room_images, room_graph = {current_room: []}, {}
    room_filenames = {current_room: []}

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % FRAME_INTERVAL != 0:
            frame_idx += 1
            continue

        emb = get_clip_embedding(frame)
        is_new_room = False

        if prev_emb is not None:
            sim = cosine_sim(emb, prev_emb)
            direction, angle = estimate_rotation(prev_frame, frame)
            is_new_room = (sim < SIM_THRESHOLD and angle >= TURN_THRESHOLD_DEGREES)
            if is_new_room:
                room_id += 1
                new_room = f"room_{room_id}"
                room_graph.setdefault(current_room, []).append({"room": new_room, "direction": direction})
                current_room = new_room
                room_images[current_room] = []
                room_filenames[current_room] = []

        # ✅ Save only clear frames
        if is_clear_image(frame):
            frame_filename = f"{OUTPUT_DIR}/{current_room}_frame_{frame_idx}.jpg"
            cv2.imwrite(frame_filename, frame)
            room_images[current_room].append(frame)
            room_filenames[current_room].append(frame_filename)

        prev_emb, prev_frame = emb, frame.copy()
        frame_idx += 1

    cap.release()

    # ✅ Save 3 start and 3 end images per room
    for room, files in room_filenames.items():
        if len(files) >= 1:
            num_start = min(3, len(files))
            num_end = min(3, len(files))

            # Start images
            for i in range(num_start):
                start_img = files[i]
                dest = os.path.join(OUTPUT_DIR, f"{room}_start_{i+1}.jpg")
                cv2.imwrite(dest, cv2.imread(start_img))

            # End images
            for i in range(num_end):
                end_img = files[-num_end + i]
                dest = os.path.join(OUTPUT_DIR, f"{room}_end_{i+1}.jpg")
                cv2.imwrite(dest, cv2.imread(end_img))

    # ✅ Save graph
    with open(f"{OUTPUT_DIR}/graph.json", "w") as f:
        json.dump(room_graph, f, indent=2)

    print("✅ Room graph:\n", json.dumps(room_graph, indent=2))
    return room_graph, room_images


# --- 🔎 Room Matching
def match_image_to_room(img, room_images):
    if isinstance(img, str):
        img = cv2.imread(img)
    img_emb = get_clip_embedding(img)
    best_score, best_room = -1, None
    for room, imgs in room_images.items():
        for ref in imgs:
            sim = cosine_sim(img_emb, get_clip_embedding(ref))
            if sim > best_score:
                best_score, best_room = sim, room
    return best_room

# --- 📍 Navigation Path (BFS)
def bfs_path(graph, start, goal):
    queue, visited = deque([(start, [])]), set()
    while queue:
        node, path = queue.popleft()
        if node == goal: return path
        for edge in graph.get(node, []):
            neighbor = edge["room"]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [(node, edge["direction"], neighbor)]))
    return []

# --- 🚀 Main
graph, room_images = build_room_graph(VIDEO_PATH)
end_room = match_image_to_room(END_IMG_PATH, room_images)
start_room = match_image_to_room(list(room_images.values())[0][0], room_images)

print(f"🎯 Destination Room: {end_room}")
print(f"🚪 Starting Room: {start_room}")

path = bfs_path(graph, start_room, end_room)
print("🧭 Navigation Path:")
for s, d, e in path:
    print(f"{s} --{d}--> {e}")
    speak(f"From {s}, go {d} to reach {e}")
