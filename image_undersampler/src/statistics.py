import numpy as np
from typing import Dict, List
from sklearn.metrics import silhouette_score
import pandas as pd
from datetime import timedelta
import os

class StatisticsGenerator:
    def __init__(self, config: Dict):
        self.config = config
    
    def calculate_statistics(self, 
                           embeddings: np.ndarray,
                           similar_groups: List[List[str]],
                           similarity_matrix: np.ndarray,
                           execution_time: float) -> Dict:
        """Calculate comprehensive statistics"""
        stats = {
            'basic': self._calculate_basic_stats(embeddings, similar_groups),
            'similarity': self._calculate_similarity_stats(similarity_matrix),
            'quality': self._calculate_quality_metrics(embeddings, similar_groups),
            'performance': self._calculate_performance_stats(execution_time)
        }
        return stats
    
    def _calculate_basic_stats(self, embeddings: np.ndarray, 
                             similar_groups: List[List[str]]) -> Dict:
        """Calculate basic statistics"""
        return {
            'n_original_samples': len(embeddings),
            'n_similar_groups': len(similar_groups),
            'avg_group_size': np.mean([len(group) for group in similar_groups]) if similar_groups else 0
        }
    
    def _calculate_similarity_stats(self, similarity_matrix: np.ndarray) -> Dict:
        """Calculate similarity statistics"""
        return {
            'mean_similarity': np.mean(similarity_matrix),
            'median_similarity': np.median(similarity_matrix),
            'max_similarity': np.max(similarity_matrix),
            'min_similarity': np.min(similarity_matrix)
        }
    
    def _calculate_quality_metrics(self, embeddings: np.ndarray, 
                                 similar_groups: List[List[str]]) -> Dict:
        """Calculate quality metrics"""
        try:
            labels = np.zeros(len(embeddings))
            for i, group in enumerate(similar_groups):
                for path_idx in group:
                    labels[path_idx] = i + 1
            silhouette = silhouette_score(embeddings, labels)
        except:
            silhouette = None
        
        return {
            'silhouette_score': silhouette
        }
    
    def _calculate_performance_stats(self, execution_time: float) -> Dict:
        """Calculate performance statistics"""
        return {
            'execution_time': execution_time,
            'execution_time_formatted': str(timedelta(seconds=int(execution_time)))
        }
    
    def generate_report(self, stats: Dict, output_dir: str):
        """Generate HTML report"""
        report_path = os.path.join(output_dir, 'undersampling_report.html')
        
        html_content = self._generate_html_report(stats)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
    
    def _generate_html_report(self, stats: Dict) -> str:
        """Generate HTML content for the report"""
        # Implementation of HTML report generation
        # (This would be a long HTML template - I can provide if needed)
        pass