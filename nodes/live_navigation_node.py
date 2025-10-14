import cv2, time, torch, numpy as np, os, json, re
from PIL import Image
from gtts import gTTS
from IPython.display import Audio, display
from transformers import CLIPProcessor, CLIPModel
from collections import deque
import playsound

# --- 📁 Config ---
url="http://192.168.1.4:5000/video_feed"
ROOM_GRAPH_PATH = "C:/Users/rishi/OneDrive/Desktop/learnings/projects/lunabot/data/room_graph/graph.json"
ROOM_IMAGES_DIR = "C:/Users/rishi/OneDrive/Desktop/learnings/projects/lunabot/data/room_graph"
DEST_IMG_PATH = "C:/Users/rishi/OneDrive/Desktop/learnings/projects/lunabot/data/ending.jpg"
VIDEO_URL = url  # Replace with your IP webcam stream
USER_START_ROOM = "room_0"  # ✅ Set this to your actual starting room
CHECK_INTERVAL = 2
SIM_THRESHOLD = 0.70
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 🔌 Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- 🔊 Text-to-Speech
# def speak(text):
#     print("🔊", text)
#     tts = gTTS(text=text, lang='en')
#     tts.save("speech.mp3")
#     display(Audio("speech.mp3", autoplay=True))

def speak(text):
    print(f"🔊 {text}")
    tts = gTTS(text=text, lang='en')
    tts.save("speech.mp3")
    playsound.playsound("speech.mp3", True)
    os.remove("speech.mp3")

# --- 🔍 Embedding + Similarity
def get_clip_embedding(img):
    inputs = clip_processor(images=Image.fromarray(img), return_tensors="pt").to(device)
    with torch.no_grad():
        return clip_model.get_image_features(**inputs)[0].cpu().numpy()

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- 📥 Load room images and infer middle
def load_room_images(directory):
    rooms = {}
    for file in os.listdir(directory):
        if not file.endswith(".jpg"): continue
        path = os.path.join(directory, file)
        parts = file.split("_")
        room = "_".join(parts[:2])
        if "_start_" in file:
            tag = "start"
        elif "_end_" in file:
            tag = "end"
        elif "_frame_" in file:
            tag = "frame"
        else:
            continue
        rooms.setdefault(room, {"start": [], "end": [], "frame": [], "middle": []})
        img = cv2.imread(path)
        if img is not None:
            rooms[room][tag].append((path, img))
    for room, sections in rooms.items():
        frame_images = sections["frame"]
        if len(frame_images) >= 3:
            sorted_frames = sorted(frame_images, key=lambda x: int(re.findall(r'frame_(\d+)', x[0])[0]))
            mid_index = len(sorted_frames) // 2
            middle_imgs = [sorted_frames[mid_index][1]]
            if mid_index + 1 < len(sorted_frames):
                middle_imgs.append(sorted_frames[mid_index + 1][1])
            rooms[room]["middle"] = middle_imgs
        rooms[room]["start"] = [img for _, img in sections["start"]]
        rooms[room]["end"] = [img for _, img in sections["end"]]
    return rooms

# --- Match one section
def match_room_section(frame, room_images, section, verbose=False):
    emb = get_clip_embedding(frame)
    best_score, best_room = -1, None
    for room, sections in room_images.items():
        for img in sections.get(section, []):
            sim = cosine_sim(emb, get_clip_embedding(img))
            if verbose:
                print(f"📏 Similarity with {room} ({section}): {sim:.4f}")
            if sim > best_score:
                best_score = sim
                best_room = room
    if best_score >= SIM_THRESHOLD:
        return best_room
    return None

# --- Path planning
def bfs_path(graph, start, goal):
    queue, visited = deque([(start, [])]), set()
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for edge in graph.get(node, []):
            neighbor = edge["room"]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [(node, edge["direction"], neighbor)]))
    return []

# --- Live navigation
def live_navigate(video_url, room_graph, room_images, dest_img, start_room):
    dest_room = match_room_section(dest_img, room_images, "end", verbose=True)

    print(f"🚪 Provided Start Room: {start_room}")
    print(f"🎯 Detected Destination Room: {dest_room}")
    if not start_room or not dest_room:
        speak("Could not detect destination room.")
        return

    path = bfs_path(room_graph, start_room, dest_room)
    if not path:
        speak("No path found from start to destination.")
        return

    print(f"🧭 Navigation path:\n{path}")
    current_step = 0
    state = "start"
    cap = cv2.VideoCapture(video_url)
    last_time = time.time()

    while cap.isOpened() and current_step < len(path):
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        now = time.time()
        if now - last_time < CHECK_INTERVAL:
            continue
        last_time = now

        current_room, direction, next_room = path[current_step]

        if state == "start":
            match = match_room_section(frame, room_images, "start", verbose=True)
            if match == current_room:
                speak(f"You are in {current_room} start. Move forward.")
                state = "middle"
        elif state == "middle":
            match = match_room_section(frame, room_images, "middle", verbose=True)
            if match == current_room:
                speak(f"You are in the middle of {current_room}. Keep going.")
                state = "end"
        elif state == "end":
            match = match_room_section(frame, room_images, "end", verbose=True)
            if match == current_room:
                speak(f"You reached the end of {current_room}. Turn {direction} to enter {next_room}.")
                current_step += 1
                state = "start"

    speak("You have reached your destination.")
    cap.release()

# --- Run
with open(ROOM_GRAPH_PATH) as f:
    graph = json.load(f)

room_images = load_room_images(ROOM_IMAGES_DIR)
dest_img = cv2.imread(DEST_IMG_PATH)

live_navigate(VIDEO_URL, graph, room_images, dest_img, USER_START_ROOM)
