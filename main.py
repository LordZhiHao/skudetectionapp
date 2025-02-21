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
import zipfile
import shutil

# SKU mapping
# SKU_MAPPING = {
#     "wraps": 1601,
#     "salads": 1101,
#     "pudding": 2706,
#     "yogurt": 2604,
#     "salad green": 11011,
#     "salad purple": 11012
# }

SKU_MAPPING = {
    "Object": 1
}

class YOLOVideoApp(QWidget):
    def __init__(self, model_path, skip_frames=5):
        super().__init__()
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.skip_frames = skip_frames
        self.frame_counter = 0
        self.last_time = time.time()

        # Add confidence threshold
        self.confidence_threshold = 0.5

        # Add debugging options
        self.debug_detections = True
        self.save_debug_images = True

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
    
        # Verify model classes at initialization
        self.verify_class_mapping()
        self.check_model_classes()

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
                        # Prepare CVAT export when video ends
                        self.prepare_cvat_upload()
                        break
                else:
                    time.sleep(0.1)

            cap.release()
        except Exception as e:
            self.log_debug(f"Error in receive_frames: {str(e)}")

    def display_frames(self):
        """Thread function to display and process frames with enhanced debugging"""
        try:
            while self.running:
                if not self.queue.empty():
                    frame = self.queue.get(timeout=1)
                    
                    # Skip frames logic
                    self.frame_counter += 1
                    if self.frame_counter % self.skip_frames != 0:
                        continue

                    # Process frame with debugging
                    print("\nProcessing frame:")
                    print(f"Frame shape: {frame.shape}")
                    
                    start_time = time.time()
                    results = self.model(frame)
                    process_time = time.time() - start_time
                    
                    print(f"Number of detections: {len(results[0].boxes)}")
                    for result in results:
                        for box in result.boxes:
                            cls = result.names[int(box.cls)]
                            conf = float(box.conf)
                            bbox = box.xyxy[0].cpu().numpy()
                            print(f"Detection: {cls}, Conf: {conf:.2f}, Box: {bbox}")

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
        try:
            base_name = f"frame_{self.frame_count:06d}"
            image_path = os.path.join(self.output_dir, "images", f"{base_name}.jpg")
            label_path = os.path.join(self.output_dir, "labels", f"{base_name}.txt")

            # Debug print
            print(f"\nProcessing results for frame {self.frame_count}")
            print(f"Results: {results}")

            # Save image
            save_success = cv2.imwrite(image_path, frame)
            if not save_success:
                self.log_debug(f"Failed to save image: {image_path}")
                return

            # Save annotations
            with open(label_path, 'w') as f:
                for result in results:
                    # Debug print
                    print(f"Number of detections: {len(result.boxes)}")
                    
                    for box in result.boxes:
                        cls = result.names[int(box.cls)]
                        confidence = float(box.conf)
                        
                        # Debug print
                        print(f"Detected class: {cls}, Confidence: {confidence}")
                        
                        # Only save if confidence is above threshold
                        if confidence > self.confidence_threshold:
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
                                # Debug print
                                print(f"Class {cls} found in mapping: {cls in SKU_MAPPING}")
                                
                                # Check if class is in mapping
                                if cls in SKU_MAPPING:
                                    class_id = list(SKU_MAPPING.keys()).index(cls)
                                    line = f"{class_id} {x_center} {y_center} {w} {h}\n"
                                    f.write(line)
                                    print(f"Wrote annotation: {line}")
                                else:
                                    self.log_debug(f"Warning: Label {cls} not found in SKU_MAPPING")

                            except Exception as e:
                                self.log_debug(f"Error processing detection: {str(e)}")

            # Save debug image if enabled
            if self.save_debug_images:
                self.save_debug_image(frame, result.boxes, self.frame_count)

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

    # picture annotations
    def check_model_classes(self):
        """Check if model classes match SKU_MAPPING"""
        model_classes = self.model.names
        print("Model classes:", model_classes)
        print("SKU mapping classes:", list(SKU_MAPPING.keys()))

    def verify_class_mapping(self):
        """Verify that model classes match SKU_MAPPING"""
        model_classes = set(self.model.names.values())
        sku_classes = set(SKU_MAPPING.keys())
        
        print("\nClass Mapping Verification:")
        print(f"Model classes: {model_classes}")
        print(f"SKU classes: {sku_classes}")
        
        missing_classes = model_classes - sku_classes
        if missing_classes:
            print(f"Warning: These classes are in the model but not in SKU_MAPPING: {missing_classes}")
        
        extra_classes = sku_classes - model_classes
        if extra_classes:
            print(f"Warning: These classes are in SKU_MAPPING but not in the model: {extra_classes}")

    def save_debug_image(self, frame, boxes, frame_number):
        """Save debug image with boxes drawn"""
        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)
        
        debug_frame = frame.copy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = self.model.names[int(box.cls)]
            conf = float(box.conf)
            
            cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"{cls}:{conf:.2f}", (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        cv2.imwrite(os.path.join(debug_dir, f"debug_frame_{frame_number}.jpg"), debug_frame)

    def prepare_cvat_files(self):
        """Prepare necessary files for CVAT YOLO format"""
        try:
            # Create cvat_dataset directory
            cvat_dataset_dir = "cvat_dataset"
            obj_train_data_dir = os.path.join(cvat_dataset_dir, "obj_train_data")
            os.makedirs(cvat_dataset_dir, exist_ok=True)
            os.makedirs(obj_train_data_dir, exist_ok=True)

            # Create obj.names file
            with open(os.path.join(cvat_dataset_dir, "obj.names"), 'w') as f:
                for class_name in SKU_MAPPING.keys():
                    f.write(f"{class_name}\n")

            # Create obj.data file
            with open(os.path.join(cvat_dataset_dir, "obj.data"), 'w') as f:
                f.write(f"classes = {len(SKU_MAPPING)}\n")
                f.write("train = train.txt\n")
                f.write("names = obj.names\n")
                f.write("backup = backup/\n")

            # Copy images and labels to obj_train_data
            images_dir = os.path.join(self.output_dir, "images")
            labels_dir = os.path.join(self.output_dir, "labels")
        
            # Create train.txt with proper paths
            with open(os.path.join(cvat_dataset_dir, "train.txt"), 'w') as f:
                for image in os.listdir(images_dir):
                    if image.endswith(('.jpg', '.jpeg', '.png')):
                        # Copy image
                        shutil.copy2(
                            os.path.join(images_dir, image),
                            os.path.join(obj_train_data_dir, image)
                        )
                    
                        # Copy corresponding label file
                        label_file = image.rsplit('.', 1)[0] + '.txt'
                        if os.path.exists(os.path.join(labels_dir, label_file)):
                            shutil.copy2(
                                os.path.join(labels_dir, label_file),
                                os.path.join(obj_train_data_dir, label_file)
                            )
                    
                        # Write to train.txt
                        f.write(f"obj_train_data/{image}\n")

            self.log_debug("Created CVAT YOLO format configuration files")

        except Exception as e:
            self.log_debug(f"Error preparing CVAT files: {str(e)}")

    def prepare_cvat_upload(self):
        """Prepare dataset for CVAT upload"""
        try:
            # First ensure all configuration files are created
            self.prepare_cvat_files()

            # Create a zip file containing everything
            dataset_zip = "cvat_dataset.zip"
            with zipfile.ZipFile(dataset_zip, 'w') as zipf:
                # Add all files from cvat_dataset directory
                for root, dirs, files in os.walk("cvat_dataset"):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, "cvat_dataset")
                        zipf.write(file_path, arcname)

            self.log_debug(f"Created {dataset_zip} for CVAT upload")

        except Exception as e:
            self.log_debug(f"Error preparing CVAT upload: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    model_path = "./finetune_yolo_data_augmentation/exp/weights/best.pt" # update model path here
    yolo_app = YOLOVideoApp(model_path)
    
    # Verify setup
    yolo_app.verify_class_mapping()
    print("\nModel information:")
    print(yolo_app.model.names)
    
    yolo_app.show()
    sys.exit(app.exec())