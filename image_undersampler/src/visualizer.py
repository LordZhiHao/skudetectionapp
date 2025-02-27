import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import cv2
import numpy as np
from typing import List, Dict
import os

class Visualizer:
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = None
        
    def set_output_dir(self, output_dir: str):
        """Set output directory for visualizations"""
        self.output_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize_similarity_matrix(self, similarity_matrix: np.ndarray):
        """Visualize similarity matrix as a heatmap"""
        plt.figure(figsize=tuple(self.config['visualization']['figure_size']))
        sns.heatmap(similarity_matrix, cmap='viridis')
        plt.title('Image Similarity Matrix')
        plt.savefig(
            os.path.join(self.output_dir, 'similarity_matrix.png'),
            dpi=self.config['visualization']['dpi']
        )
        plt.close()
    
    def visualize_similar_groups(self, similar_groups: List[List[str]]):
        """Visualize examples of similar images"""
        if not similar_groups:
            return
            
        max_groups = self.config['visualization']['max_groups_to_visualize']
        
        for i, group in enumerate(similar_groups[:max_groups]):
            n_images = len(group)
            fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
            if n_images == 1:
                axes = [axes]
                
            for ax, img_path in zip(axes, group):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(os.path.basename(img_path))
                
            plt.savefig(
                os.path.join(self.output_dir, f'similar_group_{i}.png'),
                dpi=self.config['visualization']['dpi']
            )
            plt.close()
    
    def visualize_similarity_graph(self, similarity_matrix: np.ndarray, 
                                 paths: List[str], threshold: float):
        """Visualize similarity relationships as a graph"""
        G = nx.Graph()
        
        # Add nodes
        for i, path in enumerate(paths):
            G.add_node(i, name=os.path.basename(path))
        
        # Add edges for similar images
        for i in range(len(paths)):
            for j in range(i+1, len(paths)):
                if similarity_matrix[i][j] > threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i][j])
        
        plt.figure(figsize=tuple(self.config['visualization']['figure_size']))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_size=100, node_color='lightblue',
                with_labels=False, alpha=0.7)
        
        labels = nx.get_node_attributes(G, 'name')
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.savefig(
            os.path.join(self.output_dir, 'similarity_graph.png'),
            dpi=self.config['visualization']['dpi']
        )
        plt.close()