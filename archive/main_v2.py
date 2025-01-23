import sys
import cv2
import psutil
import time
import threading
import queue
import os
from ultralytics import YOLO
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QWidget, QFileDialog
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt

# SKU mapping
SKU_MAPPING = {
    "wraps": 1601,
    "salads": 1101,
    "pudding": 2706,
    "yogurt": 2604
}

class YOLOVideoApp(QWidget):
    def __init__(self, model_path, skip_frames=10):
        super().__init__()
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.skip_frames = skip_frames
        self.frame_counter = 0
        self.last_time = time.time()

        # Queue and processing parameters
        self.max_queue_size = 30
        self.queue = queue.Queue(maxsize=self.max_queue_size)
        self.batch_size = 1
        self.running = False
        self.paused = False
        self.video_path = None

        # Thread management
        self.receive_thread = None
        self.display_thread = None

        # Debug settings
        self.debug_mode = True
        self.log_file = "debug_log.txt"

        # Output directory setup
        self.frame_count = 0
        self.output_dir = "output_dataset"
        if not os.path.exists(self.output_dir):
            os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "labels"), exist_ok=True)

        # GUI Setup
        self.setup_gui()

    def setup_gui(self):
        """Setup GUI components"""
        self.setWindowTitle("YOLO Video App")
        self.setGeometry(100, 100, 800, 600)
        self.layout = QVBoxLayout()

        # Video feed label
        self.image_label = QLabel("Video Feed")
        self.image_label.setFixedSize(640, 360)
        self.layout.addWidget(self.image_label)

        # Status labels
        self.cpu_label = QLabel("CPU Usage: 0%")
        self.ram_label = QLabel("RAM Usage: 0%")
        self.fps_label = QLabel("FPS: 0")
        self.progress_label = QLabel("Progress: 0 frames processed")
        self.queue_status_label = QLabel("Queue Status: 0/30")

        # Add labels to layout
        for label in [self.cpu_label, self.ram_label, self.fps_label, 
                     self.progress_label, self.queue_status_label]:
            self.layout.addWidget(label)

        # Buttons
        self.setup_buttons()

        self.setLayout(self.layout)

        # Timer for system stats
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(1000)

    def setup_buttons(self):
        """Setup control buttons"""
        buttons = [
            ("Upload Video", self.upload_video_dialog),
            ("Start Video", self.start_video_task),
            ("Pause/Resume Video", self.toggle_pause_task),
            ("Stop Video", self.stop_video_task)
        ]

        for text, callback in buttons:
            button = QPushButton(text)
            button.clicked.connect(callback)
            self.layout.addWidget(button)

    def log_debug(self, message):
        """Log debug messages"""
        if self.debug_mode:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a") as f:
                f.write(f"{timestamp}: {message}\n")
            print(message)

    def upload_video_dialog(self):
        """Open file dialog to select video"""
        self.video_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if self.video_path:
            self.image_label.setText(f"Loaded: {self.video_path}")
            self.log_debug(f"Video loaded: {self.video_path}")

    def receive_frames(self):
        """Thread function to read frames from video"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.log_debug(f"Total frames in video: {total_frames}")

            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue

                if self.queue.qsize() < self.max_queue_size:
                    ret, frame = cap.read()
                    if ret:
                        # Resize frame for better performance
                        frame = cv2.resize(frame, (640, 480))
                        self.queue.put(frame)
                        self.queue_status_label.setText(f"Queue Status: {self.queue.qsize()}/{self.max_queue_size}")
                    else:
                        self.log_debug("End of video reached")
                        break
                else:
                    time.sleep(0.1)

            cap.release()
        except Exception as e:
            self.log_debug(f"Error in receive_frames: {str(e)}")

    def display_frames(self):
        """Thread function to display and process frames"""
        try:
            while self.running:
                if not self.queue.empty():
                    # Monitor memory
                    memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                    self.log_debug(f"Memory usage: {memory_usage:.2f} MB")

                    frame = self.queue.get(timeout=1)
                    
                    # Skip frames logic
                    self.frame_counter += 1
                    if self.frame_counter % self.skip_frames != 0:
                        continue

                    # Process frame
                    start_time = time.time()
                    results = self.model(frame)
                    process_time = time.time() - start_time
                    self.log_debug(f"Frame processing time: {process_time:.2f} seconds")

                    # Save frame and annotations
                    self.save_frame_and_annotations(frame, results)

                    # Update display
                    self.update_display(frame, results)

                    # Update progress
                    self.update_progress()

                else:
                    time.sleep(0.1)

        except Exception as e:
            self.log_debug(f"Error in display_frames: {str(e)}")

    def save_frame_and_annotations(self, frame, results):
        """Save frame and its annotations"""
        try:
            image_path = os.path.join(self.output_dir, "images", f"frame_{self.frame_count:06d}.jpg")
            label_path = os.path.join(self.output_dir, "labels", f"frame_{self.frame_count:06d}.txt")

            # Save image
            save_success = cv2.imwrite(image_path, frame)
            if not save_success:
                self.log_debug(f"Failed to save image: {image_path}")
                return

            # Save annotations
            with open(label_path, 'w') as f:
                for result in results:
                    for box in result.boxes:
                        cls = result.names[int(box.cls)]
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        # Convert to YOLO format
                        x1, y1, x2, y2 = bbox
                        w = x2 - x1
                        h = y2 - y1
                        x_center = x1 + w/2
                        y_center = y1 + h/2

                        # Normalize coordinates
                        img_h, img_w = frame.shape[:2]
                        x_center /= img_w
                        y_center /= img_h
                        w /= img_w
                        h /= img_h

                        try:
                            class_id = list(SKU_MAPPING.keys()).index(cls)
                            f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")
                        except ValueError:
                            self.log_debug(f"Warning: Label {cls} not found in SKU_MAPPING")

            self.frame_count += 1
            self.log_debug(f"Successfully saved frame {self.frame_count}")

        except Exception as e:
            self.log_debug(f"Error saving frame {self.frame_count}: {str(e)}")

    def update_display(self, frame, results):
        """Update the display with the processed frame"""
        detection_frame = results[0].plot()
        rgb_image = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(640, 360, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def update_progress(self):
        """Update progress display"""
        self.progress_label.setText(f"Progress: {self.frame_count} frames processed")

    def start_video_task(self):
        """Start video processing"""
        if not self.running and self.video_path:
            self.running = True
            self.paused = False
            
            # Clear queue
            while not self.queue.empty():
                self.queue.get()

            self.receive_thread = threading.Thread(target=self.receive_frames, daemon=True)
            self.display_thread = threading.Thread(target=self.display_frames, daemon=True)
            
            self.receive_thread.start()
            self.display_thread.start()
            
            self.log_debug("Video processing started")
            self.log_debug(f"Receive thread alive: {self.receive_thread.is_alive()}")
            self.log_debug(f"Display thread alive: {self.display_thread.is_alive()}")

    def toggle_pause_task(self):
        """Pause or resume video processing"""
        self.paused = not self.paused
        self.log_debug(f"Video {'paused' if self.paused else 'resumed'}")

    def stop_video_task(self):
        """Stop video processing"""
        self.running = False
        self.paused = False
        self.image_label.clear()
        self.image_label.setText("Video Feed")
        self.log_debug("Video processing stopped")

    def update_stats(self):
        """Update system statistics"""
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        
        self.cpu_label.setText(f"CPU Usage: {cpu_usage}%")
        self.ram_label.setText(f"RAM Usage: {ram_usage}%")
        self.queue_status_label.setText(f"Queue Status: {self.queue.qsize()}/{self.max_queue_size}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    yolo_app = YOLOVideoApp("best.pt")
    yolo_app.show()
    sys.exit(app.exec())