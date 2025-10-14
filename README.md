# 🌕 Lunabot: Autonomous Lunar Habitat Navigator & Guardian 🚀

> A modular, ROS-style simulation in Python that *maps, navigates, detects hazards, and monitors habitat conditions* — all *without requiring ROS installation*.

---

## 🧠 Overview

As humanity prepares to build sustainable lunar habitats, robots will play a vital role in ensuring astronaut safety and operational efficiency.  
*Lunabot* brings that vision to life — a lightweight, AI-powered prototype that simulates *autonomous navigation, hazard detection, and environment monitoring* for lunar bases.

Designed in a *ROS-like modular structure*, Lunabot separates each function into independent “nodes,” making it intuitive, expandable, and ready for Smart India Hackathon demos.  

---

## 🛰 Features at a Glance

✅ *Autonomous Navigation* using graph-based BFS algorithms  
✅ *Hazard Detection* powered by YOLOv8 object detection  
✅ *Environmental Monitoring* of temperature and oxygen levels  
✅ *Voice Alerts* using Google Text-to-Speech (gTTS)  
✅ *ROS-like Architecture* for realistic simulation without heavy dependencies  

---

## 🏗 System Architecture

| Node | Purpose |
|------|----------|
| 🧱 *build_room_graph_node.py* | Builds a graph (graph.json) to represent rooms and connectivity |
| 🧭 *live_navigation_node.py* | Uses BFS to navigate between nodes and gives spoken instructions |
| 👁 *yolo_detector_node.py* | Detects hazards in video frames using YOLOv8; triggers TTS alerts |
| 🌡 *env_monitor_node.py* | Simulates environment readings (Temp, O₂) and warns on unsafe levels |
| 🚀 *lunar_nav.launch.py* | Acts as a ROS-style launch controller, orchestrating all nodes |

---

## 📁 Repository Structure



lunabot/
├── package.xml
├── launch/
│   └── lunar_nav.launch.py
├── nodes/
│   ├── build_room_graph_node.py
│   ├── live_navigation_node.py
│   ├── yolo_detector_node.py
│   └── env_monitor_node.py
├── data/
│   ├── room_graph/        ← stores frames + graph.json
│   ├── your_video.mp4     ← input for hazard detection
│   └── ending.jpg
├── requirements.txt
└── README.md

`

---

## ⚙ Installation & Setup

1. *Clone the repository*
   bash
   git clone https://github.com/Rishika-Mashetty/lunabot.git
   cd lunabot
`

2. *Create a virtual environment (recommended)*

   bash
   python -m venv venv
   source venv/bin/activate      # On macOS/Linux
   venv\Scripts\activate         # On Windows
   

3. *Install dependencies*

   bash
   pip install -r requirements.txt
   

4. *Dependencies include:*

   * torch
   * ultralytics
   * opencv-python
   * gTTS
   * IPython
   * numpy
   * Pillow
   * transformers

---

## 🚀 How to Run the Project

### Option 1 — Full Simulation Launch

Run all nodes via the launch file:

bash
python3 launch/lunar_nav.launch.py


This will:

* Build the room graph
* Start hazard detection (yolo_detector_node.py)
* Start environment monitoring (env_monitor_node.py)
* Begin navigation through habitat (live_navigation_node.py)

---

### Option 2 — Run Nodes Individually

*Build the Room Graph*

bash
python3 nodes/build_room_graph_node.py


*Start Navigation*

bash
python3 nodes/live_navigation_node.py


*Run YOLOv8 Hazard Detection*

bash
python3 nodes/yolo_detector_node.py


*Run Environment Monitor*

bash
python3 nodes/env_monitor_node.py


---

## 🎥 Demo Instructions (for Judges)

1. Open your console and run:

   bash
   python3 launch/lunar_nav.launch.py
   
2. Observe console outputs for:

   * Room mapping and navigation logs
   * Hazard detection results (object counts)
   * Environmental alerts (temperature & O₂)
3. Listen for *real-time voice feedback*:

   * “Hazard detected: 2 objects”
   * “Warning! Temperature too high: 32°C”
   * “Oxygen level low: 19.8%”
4. Optional: Record your screen and TTS audio for a 1–2 minute demo video.

---

## ✨ Key Highlights

* 🧩 *ROS-style Modular Design* — instantly recognizable structure for robotics judges
* 👁 *AI-driven Hazard Awareness* — powered by YOLOv8
* 🗺 *Graph-based Path Planning* — simple BFS route finder
* 🔊 *Voice Feedback System* — gTTS-based live alerts
* 🧪 *Lightweight Simulation* — runs entirely in Python
* 💡 *Hackathon-ready* — perfect for Smart India Hackathon or demo environments

---

## 💡 Future Enhancements

* Add *LiDAR / SLAM modules* for real-time mapping
* Integrate *sensor APIs* for live oxygen and temperature data
* Enable *multi-robot coordination* within habitat zones
* Deploy to *Raspberry Pi / Jetson Nano* for real-hardware demos
* Extend to *Mars or space-station simulations*

---

## 👩‍🚀 Team Lunabot

A passionate group of innovators exploring the intersection of *AI, robotics, and space exploration*.
Created for *Smart India Hackathon 2025* to demonstrate intelligent automation for future space habitats.

> “Lunabot — Bringing intelligence and safety to life beyond Earth.” 🌌



## 🌐 Links & Resources

* 🔗 *GitHub Repository:* [https://github.com/Rishika-Mashetty/lunabot](https://github.com/Rishika-Mashetty/lunabot)
* 🛰 *Keywords:* AI, Robotics, Space Tech, Python, YOLOv8, ROS Simulation, BFS Navigation
* 📹 *Demo Tip:* Run in VS Code terminal for voice playback and easy screen recording.

---

⭐ *If you liked this project, give it a star on GitHub and share your feedback!*

```
