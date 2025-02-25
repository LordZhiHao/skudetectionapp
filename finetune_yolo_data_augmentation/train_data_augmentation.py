import os
import cv2
import numpy as np
from pathlib import Path

def augment_dataset(base_path):
    # Define paths
    train_images_path = os.path.join(base_path, 'images', 'train')
    train_labels_path = os.path.join(base_path, 'labels', 'train')

    # Get list of original images
    image_files = [f for f in os.listdir(train_images_path) if f.endswith(('.jpg', '.jpeg', '.png', 'PNG'))]

    for image_file in image_files:
        # Read image
        image_path = os.path.join(train_images_path, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Could not read image: {image_file}")
            continue

        # Get corresponding label file
        label_file = os.path.join(train_labels_path, os.path.splitext(image_file)[0] + '.txt')
        
        if not os.path.exists(label_file):
            img_lr = cv2.flip(image, 1) # 1. Horizontal Flip (left-right)
            img_td = cv2.flip(image, 0) # 2. Vertical Flip (top-down)
            print(f"No label file found for, flipped image: {image_file}")
            continue

        # Read labels
        with open(label_file, 'r') as f:
            labels = f.readlines()

        # 1. Horizontal Flip (left-right)
        img_lr = cv2.flip(image, 1)
        labels_lr = []
        
        for label in labels:
            class_id, x, y, w, h = map(float, label.strip().split())
            # Flip x coordinate for horizontal flip
            x_lr = 1 - x
            labels_lr.append(f"{int(class_id)} {x_lr} {y} {w} {h}\n")

        # Save horizontal flip
        lr_image_path = os.path.join(train_images_path, f"{os.path.splitext(image_file)[0]}_lr{os.path.splitext(image_file)[1]}")
        lr_label_path = os.path.join(train_labels_path, f"{os.path.splitext(image_file)[0]}_lr.txt")
        
        cv2.imwrite(lr_image_path, img_lr)
        with open(lr_label_path, 'w') as f:
            f.writelines(labels_lr)

        # 2. Vertical Flip (top-down)
        img_td = cv2.flip(image, 0)
        labels_td = []
        
        for label in labels:
            class_id, x, y, w, h = map(float, label.strip().split())
            # Flip y coordinate for vertical flip
            y_td = 1 - y
            labels_td.append(f"{int(class_id)} {x} {y_td} {w} {h}\n")

        # Save vertical flip
        td_image_path = os.path.join(train_images_path, f"{os.path.splitext(image_file)[0]}_td{os.path.splitext(image_file)[1]}")
        td_label_path = os.path.join(train_labels_path, f"{os.path.splitext(image_file)[0]}_td.txt")
        
        cv2.imwrite(td_image_path, img_td)
        with open(td_label_path, 'w') as f:
            f.writelines(labels_td)

        print(f"Processed: {image_file}")

if __name__ == "__main__":
    # Specify your base directory path
    base_path = "/Users/Jason/Desktop/dataset" # Replace with your actual path
    augment_dataset(base_path)