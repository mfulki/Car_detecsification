from ultralytics import YOLO
from utils.logger import TrainingLogger

class DetectionTrainer:
    def __init__(self, model_path, data_path, project_name="vehicle"):
        self.model = YOLO(model_path)
        self.data_path = data_path
        self.project_name = project_name
        self.logger = TrainingLogger()

    def train(self, **kwargs):
        self.logger.log(f"Training started on data: {self.data_path}")
        results = self.model.train(data=self.data_path, project="runs/train", name=self.project_name, **kwargs)
        self.logger.log("Training complete.")
        return results

    def validate(self):
        self.logger.log("Running validation...")
        metrics = self.model.val()
        self.logger.log(f"Validation complete.\n Metrics: {metrics}")
        return metrics

    def export(self, format="onnx"):
        self.logger.log(f"Exporting model to {format}...")
        path = self.model.export(format=format)
        self.logger.log(f"Model exported to: {path}")
        return path

    def close_logs(self):
        self.logger.close()
