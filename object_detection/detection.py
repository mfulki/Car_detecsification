import sys
import os
import logging
from datetime import datetime
from ultralytics import YOLO

os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"logs/train_{timestamp}.log"

sys.stdout = open(log_path, 'w')
sys.stderr = sys.stdout

logging.basicConfig(level=logging.INFO)
logging.info(f"Logging started at {timestamp}")

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    
    train_results = model.train(
        data="dataset/data.yaml",
        epochs=60,             
        imgsz=640,             
        batch=16,              
        device=0,            
        workers=4,            
        patience=10,          
        project="runs/train",
        name="vehicle",
        save=True,
        save_period=5,        
        val=True   
    )

    metrics = model.val()

    logging.info(f"Training results: {train_results}")

    metrics = model.val()
    logging.info(f"Validation metrics: {metrics}")

    export_path = model.export(format="onnx")
    logging.info(f"Model exported to: {export_path}")
    
    sys.stdout.close()
    
    
