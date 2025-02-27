import torch
from ultralytics import YOLO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image
import shutil
from collections import defaultdict
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2

class SimilarityBasedUndersampler:
    def __init__(self, 
                 model_path='yolov8n.pt',
                 similarity_threshold=0.85,  # Main hyperparameter
                 random_state=42):
        """
        Initialize the Similarity-Based Undersampler
        Args:
            model_path (str): Path to YOLO model
            similarity_threshold (float): Threshold for image similarity (0.0 to 1.0)
            random_state (int): Random seed
        """
        self.setup_logging()
        self.logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        self.similarity_threshold = similarity_threshold
        np.random.seed(random_state)

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('SimilarityBasedUndersampler')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            c_handler = logging.StreamHandler()
            f_handler = logging.FileHandler(
                os.path.join(log_dir, f'undersampling_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            )
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            c_handler.setFormatter(formatter)
            f_handler.setFormatter(formatter)
            
            self.logger.addHandler(c_handler)
            self.logger.addHandler(f_handler)

    def extract_embeddings(self, image_path: str) -> Optional[np.ndarray]:
        """Extract embeddings from an image using YOLO"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Could not read image: {image_path}")
                return None
                
            results = self.model(img, verbose=False)
            features = results[0].probs.data.cpu().numpy()
            return features
        except Exception as e:
            self.logger.error(f"Error extracting embeddings from {image_path}: {str(e)}")
            return None

    def find_similar_groups(self, embeddings: np.ndarray, paths: List[str]) -> List[List[str]]:
        """
        Group similar images based on similarity threshold
        """
        similarity_matrix = cosine_similarity(embeddings)
        n_samples = len(paths)
        
        # Initialize groups
        groups = []
        processed = set()
        
        for i in range(n_samples):
            if i in processed:
                continue
                
            # Find all images similar to current image
            similar_indices = [i]
            for j in range(i + 1, n_samples):
                if j not in processed and similarity_matrix[i][j] > self.similarity_threshold:
                    similar_indices.append(j)
                    
            if len(similar_indices) > 1:  # Only create groups for similar images
                groups.append([paths[idx] for idx in similar_indices])
                processed.update(similar_indices)
            else:
                processed.add(i)
        
        return groups

    def undersample_dataset(self, input_dir: str, output_dir: str):
        """
        Undersample dataset based on similarity threshold
        """
        self.logger.info(f"Starting undersampling with similarity threshold: {self.similarity_threshold}")
        
        # Setup directories
        input_images = os.path.join(input_dir, 'images')
        input_labels = os.path.join(input_dir, 'labels')
        output_images = os.path.join(output_dir, 'images')
        output_labels = os.path.join(output_dir, 'labels')
        
        os.makedirs(output_images, exist_ok=True)
        os.makedirs(output_labels, exist_ok=True)
        
        # Get all images
        image_files = [f for f in os.listdir(input_images) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Extract embeddings
        valid_images = []
        valid_embeddings = []
        
        self.logger.info("Extracting image embeddings...")
        for img_file in image_files:
            img_path = os.path.join(input_images, img_file)
            embedding = self.extract_embeddings(img_path)
            
            if embedding is not None:
                valid_images.append(img_path)
                valid_embeddings.append(embedding)
        
        valid_embeddings = np.array(valid_embeddings)
        
        # Find similar groups
        self.logger.info("Finding similar image groups...")
        similar_groups = self.find_similar_groups(valid_embeddings, valid_images)
        
        # Process and copy images
        processed_images = set()
        removed_count = 0
        
        self.logger.info("Processing similar groups and copying files...")
        
        # First, copy all images that aren't in any similar group
        for img_path in valid_images:
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            
            if not any(img_path in group for group in similar_groups):
                shutil.copy2(img_path, os.path.join(output_images, img_name))
                label_path = os.path.join(input_labels, label_name)
                if os.path.exists(label_path):
                    shutil.copy2(label_path, os.path.join(output_labels, label_name))
                processed_images.add(img_path)
        
        # Then process similar groups
        for group in similar_groups:
            # Keep the first image from each group
            keeper = group[0]
            if keeper not in processed_images:
                img_name = os.path.basename(keeper)
                label_name = os.path.splitext(img_name)[0] + '.txt'
                
                shutil.copy2(keeper, os.path.join(output_images, img_name))
                label_path = os.path.join(input_labels, label_name)
                if os.path.exists(label_path):
                    shutil.copy2(label_path, os.path.join(output_labels, label_name))
                    
                processed_images.add(keeper)
                removed_count += len(group) - 1
        
        # Generate statistics
        stats = {
            'original_count': len(valid_images),
            'final_count': len(processed_images),
            'removed_count': removed_count,
            'similar_groups': len(similar_groups),
            'reduction_percentage': (removed_count / len(valid_images)) * 100 if valid_images else 0
        }
        
        self.logger.info("\nUndersampling Statistics:")
        self.logger.info(f"Original images: {stats['original_count']}")
        self.logger.info(f"Final images: {stats['final_count']}")
        self.logger.info(f"Removed images: {stats['removed_count']}")
        self.logger.info(f"Similar groups found: {stats['similar_groups']}")
        self.logger.info(f"Reduction percentage: {stats['reduction_percentage']:.2f}%")
        
        return stats

def main():
    # Initialize undersampler with similarity threshold
    undersampler = SimilarityBasedUndersampler(
        model_path='yolov8n.pt',
        similarity_threshold=0.85  # Adjust this value to control similarity sensitivity
    )
    
    # Define directories
    input_dir = 'OUTPUT_DATASET'
    output_dir = 'UNDERSAMPLED_DATASET'
    
    # Perform undersampling
    stats = undersampler.undersample_dataset(input_dir, output_dir)
    
    print("\nUndersampling Complete!")
    print(f"Original images: {stats['original_count']}")
    print(f"Final images: {stats['final_count']}")
    print(f"Removed {stats['removed_count']} similar images")
    print(f"Reduction: {stats['reduction_percentage']:.2f}%")

if __name__ == "__main__":
    main()