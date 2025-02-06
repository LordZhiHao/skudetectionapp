import os
import pandas as pd
import json

# At the start of your script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

def verify_images_and_annotations(annotations_df, input_dir):
    missing_files = []
    for filename in annotations_df['filename'].unique():  # adjust column name if different
        file_path = os.path.join(input_dir, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
    
    print(f"Total missing files: {len(missing_files)}")
    if missing_files:
        print("First few missing files:")
        for f in missing_files[:5]:
            print(f"- {f}")
    
    # Print some path information for debugging
    print(f"\nInput directory: {input_dir}")
    print(f"Input directory exists: {os.path.exists(input_dir)}")
    print(f"Files in input directory: {len(os.listdir(input_dir))}")

def coco_to_fastdup_annotations_with_bbox(coco_json_path):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create mappings
    image_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    category_mapping = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    annotations_list = []
    for img in coco_data['images']:
        image_id = img['id']
        filename = img['file_name']
        
        # Find all annotations for this image
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        
        for ann in image_annotations:
            bbox = ann['bbox']  # [x,y,width,height]
            category = category_mapping[ann['category_id']]
            
            # Convert COCO bbox format to xmin,ymin,xmax,ymax
            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height
            
            annotations_list.append({
                'filename': filename,
                'label': category,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
    
    df = pd.DataFrame(annotations_list)
    return df

# Run fastdup on annotated images
# Convert COCO to fastdup format
coco_path = './annotations/instances_default.json'
annotated_annotations_df = coco_to_fastdup_annotations_with_bbox(coco_path)
PATH_TO_IMAGES = './images/default' 
PATH_TO_IMAGES = os.path.abspath(PATH_TO_IMAGES)
# verify_images_and_annotations(annotated_annotations_df, PATH_TO_IMAGES)
print(annotated_annotations_df.head())
print(PATH_TO_IMAGES)

def convert_windows_path_to_wsl(windows_path):
    # Remove drive letter and convert backslashes to forward slashes
    path = windows_path.replace('\\', '/')
    # Remove drive letter (C:) and add /mnt/c
    if ':' in path:
        path = '/mnt/c' + path[2:]
    return path

# Convert input directory path
input_dir = convert_windows_path_to_wsl(r"C:\Users\Jason\Desktop\skudetectionapp\fastdup\images\default")
print(f"Converted input_dir: {input_dir}")

print("Files in directory:")
print(os.listdir(input_dir)[:5])  # First 5 files

print("\nFirst 5 filenames in annotations:")
print(annotated_annotations_df['filename'].head())

test_file = os.path.join(input_dir, annotated_annotations_df['filename'].iloc[0])
print(f"Testing file existence: {test_file}")
print(f"File exists: {os.path.exists(test_file)}")

