import os
import tempfile
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, jsonify
from ultralytics import YOLO
import threading

app = Flask(__name__)

# Directory where processed videos will be saved
DEFAULT_SAVE_DIR = r'D:\gradio'
detection_thread = None
stop_flag = threading.Event()

# YOLO models configuration
YOLO_MODELS = {
    "YOLOv8": [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"
    ],
    "YOLOv8-seg": [
        "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"
    ],
    "YOLOv8-pose": [
        "yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt", "yolov8x-pose-p6.pt"
    ],
    "YOLOv8-obb": [
        "yolov8n-obb.pt", "yolov8s-obb.pt", "yolov8m-obb.pt", "yolov8l-obb.pt", "yolov8x-obb.pt"
    ],
    "YOLOv8-cls": [
        "yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt"
    ],
    "YOLOv9": [
        "YOLOv9t", "YOLOv9s", "YOLOv9m", "YOLOv9c", "YOLOv9e"
    ],
    "YOLOv10": [
        "YOLOv10-N", "YOLOv10-S", "YOLOv10-M", "YOLOv10-B", "YOLOv10-L", "YOLOv10-X"
    ]
}

def process_video(video_path, region_points, classes_to_count, model, save_dir):
    import cv2
    from ultralytics import solutions

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error reading video file"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    processed_video_path = os.path.join(save_dir, "processed_video.mp4")
    video_writer = cv2.VideoWriter(processed_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counter = solutions.ObjectCounter(
        view_img=True,
        reg_pts=region_points,
        names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )

    while cap.isOpened() and not stop_flag.is_set():
        success, im0 = cap.read()
        if not success:
            break
        
        im0 = draw_polygon_on_frame(im0, region_points)
        tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)
        im0 = counter.start_counting(im0, tracks)
        video_writer.write(im0)
    
    cap.release()
    video_writer.release()
    
    return processed_video_path, None

def draw_polygon_on_frame(frame, polygon_points):
    import cv2
    if len(polygon_points) > 1:
        for i in range(len(polygon_points)):
            pt1 = polygon_points[i]
            pt2 = polygon_points[(i + 1) % len(polygon_points)]
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    return frame

@app.route('/')
def index():
    return render_template('index.html', message=None, models=YOLO_MODELS, default_save_dir=DEFAULT_SAVE_DIR)

@app.route('/upload', methods=['POST'])
def upload():
    global detection_thread, stop_flag
    stop_flag.clear()  # Reset the stop flag

    if 'video' not in request.files:
        return redirect(request.url)
    video_file = request.files['video']
    model_name = request.form['model']
    model_category = request.form['model_category']
    classes_input = request.form['classes']
    points_input = request.form['points']
    save_dir = request.form.get('save_dir', DEFAULT_SAVE_DIR)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        video_file.save(temp_file.name)
        temp_file_path = temp_file.name

    try:
        classes_to_count = [int(cls.strip()) for cls in classes_input.split(',') if cls.strip().isdigit()]
        points = []
        for point_str in points_input.split(';'):
            point_str = point_str.strip(' ()')
            if point_str:
                x, y = point_str.split(',')
                try:
                    points.append((int(x.strip()), int(y.strip())))
                except ValueError:
                    return "Invalid point format: Ensure all points are valid integers."

        region_points = points
        model = YOLO(os.path.join('models', model_category, model_name))  # Load the YOLO model
        
        # Process video in a separate thread to allow stopping
        detection_thread = threading.Thread(target=process_video, args=(temp_file_path, region_points, classes_to_count, model, save_dir))
        detection_thread.start()

        return render_template('index.html', message="Detection started. You can stop it if necessary.", models=YOLO_MODELS, default_save_dir=save_dir)
    except Exception as e:
        return str(e)

@app.route('/stop', methods=['POST'])
def stop_detection():
    global detection_thread, stop_flag
    stop_flag.set()  # Set the stop flag to true

    if detection_thread and detection_thread.is_alive():
        detection_thread.join()  # Wait for the detection thread to finish
        return render_template('index.html', message="Detection stopped successfully.", models=YOLO_MODELS, default_save_dir=DEFAULT_SAVE_DIR)
    else:
        return render_template('index.html', message="No active detection process.", models=YOLO_MODELS, default_save_dir=DEFAULT_SAVE_DIR)

@app.route('/downloads/<filename>')
def download_file(filename):
    try:
        return send_from_directory(directory=DEFAULT_SAVE_DIR, path=filename)
    except FileNotFoundError:
        return "File not found."

if __name__ == '__main__':
    app.run(debug=True)
