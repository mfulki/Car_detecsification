from train.detection import DetectionTrainer

def main():
    trainer = DetectionTrainer(
        model_path="yolo11n.pt",
        data_path="dataset/data.yaml",
        project_name="vehicle"
    )

    trainer.train(
        epochs=60,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=10,
        save=True,
        save_period=5,
        val=True
    )

    trainer.validate()
    trainer.export()
    # trainer.close_logs()
    
if __name__ == "__main__":
    main()
