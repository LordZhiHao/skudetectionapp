import fastdup as fd
import json
import pandas as pd
import os
import shutil

# At the start of your script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Current working directory: {os.getcwd()}")

def verify_images_and_annotations(annotations_df, input_dir):
    print("\nVerifying images and annotations...")
    print(f"Input directory: {input_dir}")
    print(f"Directory exists: {os.path.exists(input_dir)}")
    
    if os.path.exists(input_dir):
        files_in_dir = os.listdir(input_dir)
        print(f"Number of files in directory: {len(files_in_dir)}")
        print("First 5 files in directory:", files_in_dir[:5])
    
    print("\nFirst 5 rows of annotations:")
    print(annotations_df.head())
    
    missing_files = []
    for filename in annotations_df['filename'].unique():
        file_path = os.path.join(input_dir, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
    
    print(f"\nTotal missing files: {len(missing_files)}")
    if missing_files:
        print("First 5 missing files:")
        for f in missing_files[:5]:
            print(f"- {f}")

def coco_to_fastdup_annotations_with_bbox(coco_json_path):
    print(f"\nReading COCO annotations from: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"Number of images in COCO: {len(coco_data['images'])}")
    print(f"Number of annotations in COCO: {len(coco_data['annotations'])}")
    
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
    print(f"\nCreated annotations DataFrame with {len(df)} rows")
    return df

# Use the WSL local path instead of Windows path
input_dir = os.path.expanduser('~/fastdup_images')
print(f"Using WSL local path: {input_dir}")

# Clear existing output directory
output_dir = 'fastdup_output'
if os.path.exists(output_dir):
    print(f"\nRemoving existing output directory: {output_dir}")
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Convert COCO to fastdup format
coco_path = './annotations/instances_default.json'
print(f"\nProcessing COCO annotations from: {coco_path}")

# Get annotations
annotations_df = coco_to_fastdup_annotations_with_bbox(coco_path)
annotations_df.to_csv('fastdup_annotations_with_bbox.csv', index=False)

# Verify images and annotations before running
verify_images_and_annotations(annotations_df, input_dir)

try:
    # Initialize fastdup
    fd = fd.create(work_dir=output_dir)

    # Run fastdup
    print("\nRunning fastdup...")
    fd.run(
        input_dir=input_dir,
        threshold=0.7,
        lower_threshold=0.3,
        nearest_neighbors_k=2,
        distance='cosine',
        batch_size=64
    )

    # Explore results
    print("\nStarting exploration server...")
    fd.explore()

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    print("\nDirectory contents:")
    print(os.listdir(input_dir))
    raise

# fastdup_params = {
#     'nn_provider', 
#     'batch_size', 
#     'threshold', 
#     'lower_threshold', 
#     'min_offset', 
#     'num_images', 
#     'work_dir', 
#     'nnf_mode', 
#     'max_offset', 
#     'version', 
#     'model_path', 
#     'verbose', 
#     'bounding_box', 
#     'distance', 
#     'high_accuracy', 
#     'num_threads', 
#     'run_mode', 
#     'nearest_neighbors_k', 
#     'd', 
#     'resume', 
#     'input_dir', 
#     'nnf_param', 
#     'test_dir', 
#     'compute', 
#     'license'
# }

# for installation and running fastdup
# sudo apt update
# sudo apt -y install software-properties-common
# sudo add-apt-repository -y ppa:deadsnakes/ppa
# sudo apt update
# sudo apt -y install python3.9
# sudo apt -y install python3-pip
# sudo apt -y install libgl1-mesa-glx
# pip3 install --upgrade pip
# python3.9 -m pip install fastdup
# in case of cant find images 
# In WSL terminal
# mkdir -p ~/fastdup_images
# cp -r /mnt/c/Users/Jason/Desktop/skudetectionapp/fastdup/images/default/* ~/fastdup_images/
# ls -la ~/fastdup_images  # Verify files were copied
# to run
# in wsl run python3.9 /mnt/c/Users/Jason/Desktop/skudetectionapp/fastdup/main.py