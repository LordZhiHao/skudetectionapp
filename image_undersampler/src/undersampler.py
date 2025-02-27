from ultralytics import YOLO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import cv2
from PIL import Image
import os
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import gc
from .visualizer import Visualizer
from .statistics import StatisticsGenerator
from .utils import setup_logging, memory_stats
import time
import shutil

class SimilarityBasedUndersampler:
    def __init__(self, config: Dict):
        """Initialize the undersampler with configuration"""
        self.config = config
        self.logger = setup_logging(level=config['output']['log_level'])
        self.visualizer = Visualizer(config)
        self.statistics = StatisticsGenerator(config)
        self.similarity_threshold = config['undersampling']['similarity_threshold']
        self.max_size = config['undersampling']['max_image_size']
        self.batch_size = config['model']['batch_size']
        
        self._setup_model()
    
    def _setup_model(self):
        """Setup YOLO model"""
        try:
            self.model = YOLO(self.config['model']['path'])
            self.model.fuse()
            self.model.eval()
            self.model.to(self.config['model']['device'])
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image with error handling"""
        try:
            # Try cv2 first
            img = cv2.imread(image_path)
            if img is None:
                # Try PIL if cv2 fails
                try:
                    pil_img = Image.open(image_path)
                    img = np.array(pil_img)
                    if len(img.shape) == 2:  # Grayscale
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 4:  # RGBA
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                except Exception as e:
                    self.logger.error(f"Failed to load image with PIL: {str(e)}")
                    return None
                    
            # Ensure image is in correct format
            if img.shape[2] != 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            # Resize if too large
            if max(img.shape) > self.max_size:
                scale = self.max_size / max(img.shape)
                img = cv2.resize(img, None, fx=scale, fy=scale)
                
            return img
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    def extract_embeddings(self, image_path: str) -> Optional[np.ndarray]:
        """Extract embeddings from an image using YOLO"""
        try:
            # Load and preprocess image
            img = self.load_and_preprocess_image(image_path)
            if img is None:
                return None
                
            # Get model predictions
            results = self.model(img, verbose=False)
            
            # Try multiple methods to get features
            features = None
            
            if hasattr(results[0], 'features'):
                features = results[0].features.cpu().numpy()
            elif hasattr(results[0], 'boxes') and len(results[0].boxes.data) > 0:
                features = results[0].boxes.data.cpu().numpy()
            elif hasattr(results[0], 'probs') and results[0].probs is not None:
                features = results[0].probs.data.cpu().numpy()
            
            if features is None:
                self.logger.warning(f"No features extracted from {image_path}")
                return None
                
            # Ensure features are properly shaped
            if features.size == 0:
                return None
                
            # Reshape based on dimensionality
            if features.ndim == 1:
                features = features.reshape(1, -1)
            elif features.ndim > 2:
                features = features.reshape(features.shape[0], -1)
                features = np.mean(features, axis=0)
                features = features.reshape(1, -1)
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting embeddings from {image_path}: {str(e)}")
            return None

    def batch_process_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """Process images in batches to optimize memory usage"""
        embeddings = []
        valid_paths = []
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_embeddings = []
            
            for path in batch_paths:
                embedding = self.extract_embeddings(path)
                if embedding is not None:
                    batch_embeddings.append(embedding)
                    valid_paths.append(path)
            
            if batch_embeddings:
                embeddings.extend(batch_embeddings)
                
            # Optimize memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return embeddings, valid_paths

    def find_similar_groups(self, embeddings: np.ndarray, paths: List[str]) -> List[List[str]]:
        """Group similar images based on similarity threshold"""
        if len(embeddings) == 0:
            return []
            
        try:
            # Ensure embeddings is 2D
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
                
            similarity_matrix = cosine_similarity(embeddings)
            n_samples = len(paths)
            
            groups = []
            processed = set()
            
            for i in range(n_samples):
                if i in processed:
                    continue
                    
                similar_indices = [i]
                for j in range(i + 1, n_samples):
                    if j not in processed and similarity_matrix[i][j] > self.similarity_threshold:
                        similar_indices.append(j)
                        
                if len(similar_indices) > 1:
                    groups.append([paths[idx] for idx in similar_indices])
                    processed.update(similar_indices)
                else:
                    processed.add(i)
            
            return groups, similarity_matrix
            
        except Exception as e:
            self.logger.error(f"Error in find_similar_groups: {str(e)}")
            return [], None

    def copy_files(self, img_path: str, input_labels: str, output_images: str, output_labels: str):
        """Copy image and its corresponding label"""
        try:
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            
            shutil.copy2(img_path, os.path.join(output_images, img_name))
            label_path = os.path.join(input_labels, label_name)
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(output_labels, label_name))
        except Exception as e:
            self.logger.error(f"Error copying files for {img_path}: {str(e)}")

    def undersample_dataset(self, input_dir: str, output_dir: str) -> Dict:
        """Main method to undersample the dataset"""
        start_time = time.time()
        self.logger.info(f"Starting undersampling with similarity threshold: {self.similarity_threshold}")
        
        # Log initial memory usage
        mem_stats = memory_stats()
        self.logger.info(f"Initial memory usage: {mem_stats['rss_mb']:.2f} MB")
        
        # Setup directories
        input_images = os.path.join(input_dir, 'images')
        input_labels = os.path.join(input_dir, 'labels')
        output_images = os.path.join(output_dir, 'images')
        output_labels = os.path.join(output_dir, 'labels')
        
        os.makedirs(output_images, exist_ok=True)
        os.makedirs(output_labels, exist_ok=True)
        self.visualizer.set_output_dir(output_dir)
        
        # Get all images
        image_files = [f for f in os.listdir(input_images) 
                      if f.lower().endswith(tuple(self.config['undersampling']['supported_formats']))]
        
        if not image_files:
            self.logger.warning("No images found in input directory")
            return self._empty_stats()
        
        # Process images in batches
        image_paths = [os.path.join(input_images, f) for f in image_files]
        embeddings, valid_paths = self.batch_process_images(image_paths)
        
        if not valid_paths:
            self.logger.warning("No valid embeddings could be extracted")
            return self._empty_stats(total_images=len(image_files))
        
        # Convert to numpy array and find similar groups
        embeddings = np.vstack(embeddings)
        similar_groups, similarity_matrix = self.find_similar_groups(embeddings, valid_paths)
        
        # Generate visualizations if enabled
        if self.config['output']['save_visualizations']:
            self.visualizer.visualize_similarity_matrix(similarity_matrix)
            self.visualizer.visualize_similar_groups(similar_groups)
            self.visualizer.visualize_similarity_graph(similarity_matrix, valid_paths, 
                                                     self.similarity_threshold)
        
        # Process and copy images
        processed_images = set()
        removed_count = 0
        
        # Copy unique images
        self.logger.info("Copying unique images...")
        for img_path in tqdm(valid_paths, desc="Processing images"):
            if not any(img_path in group for group in similar_groups):
                self.copy_files(img_path, input_labels, output_images, output_labels)
                processed_images.add(img_path)
        
        # Process similar groups
        self.logger.info("Processing similar groups...")
        for group in tqdm(similar_groups, desc="Processing groups"):
            keeper = group[0]
            if keeper not in processed_images:
                self.copy_files(keeper, input_labels, output_images, output_labels)
                processed_images.add(keeper)
                removed_count += len(group) - 1
        
        # Calculate statistics
        execution_time = time.time() - start_time
        stats = self.statistics.calculate_statistics(
            embeddings=embeddings,
            similar_groups=similar_groups,
            similarity_matrix=similarity_matrix,
            execution_time=execution_time
        )
        
        # Generate report if enabled
        if self.config['output']['generate_report']:
            self.statistics.generate_report(stats, output_dir)
        
        # Log final memory usage
        mem_stats = memory_stats()
        self.logger.info(f"Final memory usage: {mem_stats['rss_mb']:.2f} MB")
        
        return stats

    def _empty_stats(self, total_images: int = 0) -> Dict:
        """Return empty statistics dictionary"""
        return {
            'basic': {
                'n_original_samples': total_images,
                'n_similar_groups': 0,
                'avg_group_size': 0
            },
            'similarity': {
                'mean_similarity': 0,
                'median_similarity': 0,
                'max_similarity': 0,
                'min_similarity': 0
            },
            'quality': {
                'silhouette_score': None
            },
            'performance': {
                'execution_time': 0,
                'execution_time_formatted': '0:00:00'
            }
        }