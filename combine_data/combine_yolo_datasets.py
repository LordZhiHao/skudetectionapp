import os
import shutil
import argparse
from pathlib import Path
import random
from collections import defaultdict

def create_output_structure(output_dir):
    """Create the output directory structure."""
    dirs = [
        os.path.join(output_dir, 'train', 'images'),
        os.path.join(output_dir, 'train', 'labels'),
        os.path.join(output_dir, 'val', 'images'),
        os.path.join(output_dir, 'val', 'labels')
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def combine_datasets(input_dirs, output_dir, copy_classes=True, shuffle=True, seed=42):
    """
    Combine multiple YOLO datasets into a single dataset with filename conflict resolution.
    
    Args:
        input_dirs: List of input dataset directories
        output_dir: Output directory for the combined dataset
        copy_classes: Whether to copy classes.txt from the first dataset
        shuffle: Whether to shuffle files when combining
        seed: Random seed for reproducibility
    """
    if shuffle:
        random.seed(seed)
    
    # Copy classes.txt if requested
    if copy_classes and os.path.exists(os.path.join(input_dirs[0], 'classes.txt')):
        shutil.copy(
            os.path.join(input_dirs[0], 'classes.txt'),
            os.path.join(output_dir, 'classes.txt')
        )
        print(f"Copied classes.txt from {input_dirs[0]}")
    
    # Track used filenames to avoid conflicts
    used_filenames = defaultdict(int)
    
    # Process train and val sets
    for subset in ['train', 'val']:
        # Collect all files from all datasets
        all_files = []
        
        for dataset_idx, input_dir in enumerate(input_dirs):
            dataset_name = os.path.basename(input_dir)
            subset_img_dir = os.path.join(input_dir, subset, 'images')
            subset_lbl_dir = os.path.join(input_dir, subset, 'labels')
            
            if not os.path.exists(subset_img_dir) or not os.path.exists(subset_lbl_dir):
                print(f"Warning: {subset} directories not found in {input_dir}")
                continue
            
            # Get all image files
            img_files = list(Path(subset_img_dir).glob('*.*'))
            img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.PNG']]
            
            # Get all label files
            lbl_files = list(Path(subset_lbl_dir).glob('*.txt'))
            
            # Create mapping of base names to full paths
            img_map = {f.stem: f for f in img_files}
            lbl_map = {f.stem: f for f in lbl_files}
            
            # Find files that have both image and label
            common_names = set(img_map.keys()) & set(lbl_map.keys())
            
            # Add to our collection
            for name in common_names:
                all_files.append({
                    'name': name,
                    'img_path': img_map[name],
                    'lbl_path': lbl_map[name],
                    'dataset_name': dataset_name,
                    'dataset_idx': dataset_idx
                })
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(all_files)
        
        # Copy files to output directory with conflict resolution
        for file_info in all_files:
            base_name = file_info['name']
            img_path = file_info['img_path']
            lbl_path = file_info['lbl_path']
            dataset_name = file_info['dataset_name']
            
            # Create a unique filename
            unique_name = f"{dataset_name}_{base_name}"
            
            # If this name is already used, add a counter
            if unique_name in used_filenames:
                used_filenames[unique_name] += 1
                unique_name = f"{unique_name}_{used_filenames[unique_name]}"
            else:
                used_filenames[unique_name] = 0
            
            # Copy image
            img_ext = img_path.suffix
            dst_img_path = os.path.join(output_dir, subset, 'images', f"{unique_name}{img_ext}")
            shutil.copy(img_path, dst_img_path)
            
            # Copy label
            dst_lbl_path = os.path.join(output_dir, subset, 'labels', f"{unique_name}.txt")
            shutil.copy(lbl_path, dst_lbl_path)
        
        print(f"Copied {len(all_files)} {subset} images and labels")

def main():
    parser = argparse.ArgumentParser(description='Combine multiple YOLO datasets into one')
    parser.add_argument('--input_dirs', nargs='+', required=True, 
                        help='List of input dataset directories')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory for the combined dataset')
    parser.add_argument('--no_copy_classes', action='store_false', dest='copy_classes',
                        help='Do not copy classes.txt from the first dataset')
    parser.add_argument('--no_shuffle', action='store_false', dest='shuffle',
                        help='Do not shuffle files when combining')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory structure
    create_output_structure(args.output_dir)
    
    # Combine datasets
    combine_datasets(
        args.input_dirs, 
        args.output_dir, 
        args.copy_classes,
        args.shuffle,
        args.seed
    )
    
    print("Datasets combined successfully!")

if __name__ == "__main__":
    main()