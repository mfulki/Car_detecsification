from ultralytics import YOLO



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

    metrics = model.val()
    export_path = model.export(format="onnx")
    
 
    