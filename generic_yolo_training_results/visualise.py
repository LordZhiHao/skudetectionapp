import os
import random
import cv2
import numpy as np
from ultralytics import YOLO
import time
import gc
import psutil

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    print(f"Current memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def plot_boxes(image, boxes, labels, color=(0, 255, 0), pred_color=(0, 0, 255)):
    """Plot boxes on image"""
    if len(boxes) == 0:
        return
        
    for box in boxes:
        try:
            if len(box) >= 6:  # Prediction boxes
                x1, y1, x2, y2, conf, cls = box
                c = pred_color
                label_text = f'Class {int(cls)} ({conf:.2f})'
            else:  # Ground truth boxes
                x, y, w, h = box[1:5]
                h_img, w_img = image.shape[:2]
                x1 = int((x - w/2) * w_img)
                x2 = int((x + w/2) * w_img)
                y1 = int((y - h/2) * h_img)
                y2 = int((y + h/2) * h_img)
                cls = box[0]
                c = color
                label_text = f'Class {int(cls)}'
            
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
            cv2.putText(image, label_text, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        except Exception as e:
            print(f"Error plotting box: {e}")
            continue

def process_single_image(model, img_path, label_path, index, total):
    """Process a single image and return visualization"""
    try:
        # Load and resize image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            return None

        # Resize image to reduce memory usage
        max_size = 640
        height, width = image.shape[:2]
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        # Read ground truth labels
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.readlines()
                gt_boxes = np.array([line.strip().split() for line in content], dtype=float) if content else []

        # Make prediction
        results = model(image)
        pred_boxes = results[0].boxes.data.cpu().numpy()

        # Create visualization
        img_vis = image.copy()
        
        # Plot boxes
        plot_boxes(img_vis, gt_boxes, "Ground Truth", color=(0, 255, 0))
        plot_boxes(img_vis, pred_boxes, "Prediction", pred_color=(0, 0, 255))
        
        # Add legend and status
        cv2.putText(img_vis, 'Green: Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        cv2.putText(img_vis, 'Red: Prediction', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)
        
        # Add counts
        h_img = img_vis.shape[0]
        gt_text = f"Ground Truth: {len(gt_boxes)} objects"
        pred_text = f"Prediction: {len(pred_boxes)} objects"
        cv2.putText(img_vis, gt_text, (10, h_img - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        cv2.putText(img_vis, pred_text, (10, h_img - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)

        return img_vis

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def main():
    try:
        # Set paths with your specific directories
        val_images_dir = '/Users/Jason/Desktop/dataset/images/val'
        val_labels_dir = '/Users/Jason/Desktop/dataset/labels/val'
        model_path = 'best.pt'  # Assuming best.pt is in current directory
        output_dir = '/Users/Jason/Desktop/skudetectionapp/generic_yolo_training_results/output_visualizations'  # Output directory in your dataset folder
        
        # Verify directories exist
        if not os.path.exists(val_images_dir):
            raise Exception(f"Images directory not found: {val_images_dir}")
        if not os.path.exists(val_labels_dir):
            raise Exception(f"Labels directory not found: {val_labels_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load model
        print("Loading YOLO model...")
        model = YOLO(model_path)
        model.cpu()
        print("Model loaded successfully")
        print_memory_usage()

        # Get image files
        image_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.jpeg', '.png', 'PNG'))]
        if not image_files:
            raise Exception(f"No images found in {val_images_dir}")
            
        num_samples = min(10, len(image_files))
        selected_images = random.sample(image_files, num_samples)

        print(f"Processing {num_samples} images...")

        for i, img_file in enumerate(selected_images):
            print(f"\nProcessing image {i+1}/{num_samples}: {img_file}")
            print_memory_usage()

            img_path = os.path.join(val_images_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(val_labels_dir, label_file)

            # Process image
            img_vis = process_single_image(model, img_path, label_path, i+1, num_samples)

            if img_vis is not None:
                # Save visualization
                output_path = os.path.join(output_dir, f'visualization_{i+1}.jpg')
                cv2.imwrite(output_path, img_vis)
                print(f"Saved visualization to {output_path}")

                # Display image
                cv2.imshow(f'Image {i+1}/{num_samples}', img_vis)
                key = cv2.waitKey(1000)  # Show for 1 second
                cv2.destroyAllWindows()

                # Clean up
                del img_vis
                gc.collect()
                print_memory_usage()

                if key == ord('q'):
                    break

            # Force cleanup
            gc.collect()
            time.sleep(1)  # Give system time to cleanup

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        cv2.destroyAllWindows()
        if 'model' in locals():
            del model
        gc.collect()
        print("Processing completed")
        print_memory_usage()

if __name__ == "__main__":
    main()