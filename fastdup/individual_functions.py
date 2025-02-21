import fastdup
import json
import os

# Load COCO annotations
with open('path_to_your_cvat_export.json', 'r') as f:
    coco_data = json.load(f)

# Create a mapping file for fastdup
with open('annotations.csv', 'w') as f:
    f.write('filename,label\n')  # header
    for image in coco_data['images']:
        # Find annotations for this image
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image['id']]
        
        # Get category names for annotations
        category_names = []
        for ann in image_annotations:
            category = next(cat for cat in coco_data['categories'] if cat['id'] == ann['category_id'])
            category_names.append(category['name'])
        
        # Write to CSV
        f.write(f"{image['file_name']},{','.join(category_names)}\n")

        # Initialize fastdup
        fd = fastdup.create(work_dir='fastdup_output')

        # Run fastdup on your image directory
        fd.run(
            input_dir='path_to_your_images_folder',
            annotations='annotations.csv'
        )

        # Generate visualization
        fd.vis.create_components_gallery()

        # Show similar images
        fd.vis.create_duplicates_gallery()

        # Show outliers
        fd.vis.create_outliers_gallery()

        # Show statistics
        fd.vis.create_stats_gallery()

        # additional details
        fd.vis.create_components_gallery(
        num_images=100,
        conf_threshold=0.8,
        annotation_column_name='label'
        )