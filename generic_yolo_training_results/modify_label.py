import os

# Define the paths
label_paths = ['/Users/Jason/Desktop/dataset/labels/train', '/Users/Jason/Desktop/dataset/labels/val']

# Process each directory
for label_path in label_paths:
    # Get all txt files in the directory
    txt_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    
    # Process each txt file
    for txt_file in txt_files:
        file_path = os.path.join(label_path, txt_file)
        
        # Read the file content
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Process each line
        modified_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts:  # Check if line is not empty
                parts[0] = '0'  # Replace first number with '1'
                modified_lines.append(' '.join(parts) + '\n')
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.writelines(modified_lines)
        
        print(f"Processed: {txt_file}")

print("All files have been processed!")