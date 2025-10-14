import os

def generate_launch_description():
    print("Launching Lunar Habitat Navigation Prototype...")
    os.system("python3 nodes/build_room_graph_node.py")
    os.system("python3 nodes/yolo_detector_node.py &")  # Run hazard detection in background
    os.system("python3 nodes/env_monitor_node.py &")    # Run env monitor in background
    os.system("python3 nodes/live_navigation_node.py")
    return []
