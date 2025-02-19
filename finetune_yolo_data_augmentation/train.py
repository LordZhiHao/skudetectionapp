# train.py
from ultralytics import YOLO
import torch

def train_yolo():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model

    config = {
    # define dataset
    'data': 'data.yaml',
    # configure model (key settings)
    'epochs': 100,
    'imgsz': 640,
    'batch': 32,  
    'patience': 20,
    'optimizer': 'Adam',
    'lr0': 0.001,  # Reduced for stability
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    # for errors check 
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    # for data augmentation 
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 10.0,
    'perspective': 0.0001,
    'flipud': 0.5,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.3,  # Reduced
    'copy_paste': 0.3,  # Reduced
    'auto_augment': True,
    'erasing': 0.4,
    # for more project configuration
    'project': 'runs/train',
    'name': 'exp',
    'exist_ok': True,
    'pretrained': True,
    'amp': True,
    'device': 0 if torch.cuda.is_available() else 'cpu'
    }

    # Train the model
    results = model.train(**config)
    
    # Validate the model
    results = model.val()
    
    # Export the model
    success = model.export(format='onnx')

if __name__ == "__main__":
    train_yolo()