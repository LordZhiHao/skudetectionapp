import torch
from ultralytics import YOLO
import cv2
import time
import numpy as np
from datetime import datetime

def evaluate_model(model_path, video_path, save_dir='results'):
    # Load the model
    model = YOLO(model_path)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize metrics
    inference_times = []
    detections_per_frame = []
    
    # Initialize video writer for saving annotated frames
    output_video = cv2.VideoWriter(
        f'{save_dir}/annotated_video.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # Inference
        t1 = time.time()
        results = model(frame)
        t2 = time.time()
        
        inference_times.append(t2 - t1)
        
        # Get detections
        result = results[0]
        detections_per_frame.append(len(result.boxes))
        
        # Draw detections on frame
        annotated_frame = results[0].plot()
        output_video.write(annotated_frame)
        
        # Save sample frames (e.g., every 100th frame)
        if frame_count % 100 == 0:
            cv2.imwrite(f'{save_dir}/frame_{frame_count}.jpg', annotated_frame)
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    metrics = {
        'dataset_info': {
            'video_name': video_path,
            'total_frames': total_frames,
            'resolution': f'{frame_width}x{frame_height}'
        },
        'inference_results': {
            'fps': frame_count / total_time,
            'avg_inference_time': np.mean(inference_times) * 1000,  # Convert to ms
            'avg_detections_per_frame': np.mean(detections_per_frame)
        }
    }
    
    # Run validation on the video frames
    val_results = model.val()
    
    metrics['validation_results'] = {
        'precision': val_results.results_dict['metrics/precision(B)'],
        'recall': val_results.results_dict['metrics/recall(B)'],
        'mAP50': val_results.results_dict['metrics/mAP50(B)'],
        'mAP50-95': val_results.results_dict['metrics/mAP50-95(B)']
    }
    
    # Save metrics to file
    with open(f'{save_dir}/evaluation_results.txt', 'w') as f:
        f.write("2.1 Dataset Information\n")
        f.write(f"• Test Dataset Used: {metrics['dataset_info']['video_name']}\n")
        f.write(f"• Number of Images Tested: {metrics['dataset_info']['total_frames']}\n")
        f.write(f"• Resolution of Images Used for Testing: {metrics['dataset_info']['resolution']}\n\n")
        
        f.write("2.2 Inference Results\n")
        f.write("Metric\tValue\n")
        f.write(f"FPS (Frames per Second)\t{metrics['inference_results']['fps']:.2f}\n")
        f.write(f"Average Inference Time (ms)\t{metrics['inference_results']['avg_inference_time']:.2f}\n")
        f.write(f"Precision (%)\t{metrics['validation_results']['precision']*100:.2f}\n")
        f.write(f"Recall (%)\t{metrics['validation_results']['recall']*100:.2f}\n")
        f.write(f"mAP@50 (%)\t{metrics['validation_results']['mAP50']*100:.2f}\n")
        f.write(f"mAP@50-95 (%)\t{metrics['validation_results']['mAP50-95']*100:.2f}\n")
        f.write(f"Average Detections per Frame\t{metrics['inference_results']['avg_detections_per_frame']:.2f}\n")
    
    cap.release()
    output_video.release()
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model weights (best.pt)')
    parser.add_argument('--video', type=str, required=True, help='Path to test video')
    parser.add_argument('--save-dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    import os
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run evaluation
    metrics = evaluate_model(args.model, args.video, args.save_dir)
    print("Evaluation complete. Results saved in", args.save_dir)