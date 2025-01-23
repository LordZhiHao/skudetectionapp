with open('train.txt', 'r') as file:
    lines = file.readlines()

with open('train.txt', 'w') as file:
    for line in lines:
        line = line.strip()  # Remove any whitespace including \n
        if not line.startswith('obj_train_data/'):
            line = 'obj_train_data/' + line + '.jpg\n'  # Add .jpg and \n at the end
        else:
            line = line + '.jpg\n'  # Just add .jpg and \n if already has obj_train_data/
        file.write(line)