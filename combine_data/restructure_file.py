import os
import json
import random
import shutil
from pathlib import Path
import argparse

def create_directories(base_dir):
    """Create the necessary directory structure."""
    dirs = [
        os.path.join(base_dir, 'train', 'images'),
        os.path.join(base_dir, 'train', 'labels'),
        os.path.join(base_dir, 'val', 'images'),
        os.path.join(base_dir, 'val', 'labels')
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def coco_to_yolo(coco_annotation, image_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Convert COCO format annotations to YOLO format and split into train/val sets.
    
    Args:
        coco_annotation: Path to COCO JSON annotation file
        image_dir: Directory containing the images
        output_dir: Output directory for the YOLO format dataset
        train_ratio: Ratio of images to use for training (default: 0.8)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Load COCO annotations
    with open(coco_annotation, 'r') as f:
        coco_data = json.load(f)
    
    # Create category ID to index mapping (YOLO uses indices starting from 0)
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    print(f"Categories mapping: {categories}")
    
    # Create image ID to file name mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image ID
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Get list of all image IDs with annotations
    image_ids = list(annotations_by_image.keys())
    random.shuffle(image_ids)
    
    # Split into train and validation sets
    split_idx = int(len(image_ids) * train_ratio)
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]
    
    print(f"Total images with annotations: {len(image_ids)}")
    print(f"Training images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")
    
    # Process training set
    process_set(train_ids, image_info, annotations_by_image, categories, 
                image_dir, os.path.join(output_dir, 'train'))
    
    # Process validation set
    process_set(val_ids, image_info, annotations_by_image, categories, 
                image_dir, os.path.join(output_dir, 'val'))
    
    # Create a classes.txt file
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for cat in sorted(coco_data['categories'], key=lambda x: categories[x['id']]):
            f.write(f"{cat['name']}\n")

def process_set(image_ids, image_info, annotations_by_image, categories, image_dir, output_dir):
    """Process a set of images (train or val) and convert annotations to YOLO format."""
    for img_id in image_ids:
        # Get image information
        img_info = image_info[img_id]
        img_file = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Extract just the filename without any directory structure
        img_filename = os.path.basename(img_file)
        
        # Copy image to output directory
        # Look for the image in the image_dir, considering potential subdirectories
        src_img_path = os.path.join(image_dir, img_file)
        if not os.path.exists(src_img_path):
            # Try looking for the file directly in image_dir
            src_img_path = os.path.join(image_dir, img_filename)
            if not os.path.exists(src_img_path):
                print(f"Warning: Image {img_file} not found at {src_img_path}")
                continue
        
        dst_img_path = os.path.join(output_dir, 'images', img_filename)
        
        try:
            shutil.copy(src_img_path, dst_img_path)
        except Exception as e:
            print(f"Error copying {src_img_path} to {dst_img_path}: {e}")
            continue
        
        # Create YOLO annotation file
        base_name = os.path.splitext(img_filename)[0]
        yolo_annotation_path = os.path.join(output_dir, 'labels', f"{base_name}.txt")
        
        with open(yolo_annotation_path, 'w') as f:
            for ann in annotations_by_image[img_id]:
                # Get category index (YOLO class id)
                cat_id = ann['category_id']
                yolo_cat_id = categories[cat_id]
                
                # Get bounding box coordinates
                bbox = ann['bbox']  # [x, y, width, height] in COCO format
                
                # Convert to YOLO format: [x_center, y_center, width, height] normalized
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                
                # Write to file
                f.write(f"{yolo_cat_id} {x_center} {y_center} {width} {height}\n")

def main():
    parser = argparse.ArgumentParser(description='Convert COCO format to YOLO format with train/val split')
    parser.add_argument('--coco_annotation', type=str, required=True, help='Path to COCO JSON annotation file')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the YOLO format dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training data (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create directory structure
    create_directories(args.output_dir)
    
    # Convert and split the dataset
    coco_to_yolo(args.coco_annotation, args.image_dir, args.output_dir, args.train_ratio, args.seed)
    
    print("Conversion and dataset splitting completed successfully!")

if __name__ == "__main__":
    main()

# directory structure
# --coco_annotation - 
# ./salads_dataset/annotations/instances_default.json 
# ./sandwiches_dataset/annotations/instances_default.json 
# ./wraps_dataset/annotations/instances_default.json

# --image_dir - 
# ./salads_dataset/images/default/fastdup_cam5_salads_20022025/images 
# ./sandwiches_dataset/images/default/fastdup_sandwiches_cam5_21022025/fastdup 
# ./wraps_dataset/images/default/cam5_wraps_25022025_fastdup

# --output_dir - 
# ./salads_yolo_format 
# ./sandwiches_yolo_format 
# ./wraps_yolo_format