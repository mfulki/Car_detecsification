from ultralytics import YOLO

class DetectionTrainer:
    def __init__(self, model_path, data_path, project_name="vehicle"):
        self.model = YOLO(model_path)
        self.data_path = data_path
        self.project_name = project_name

    def train(self, **kwargs):
        results = self.model.train(data=self.data_path, project="runs/train", name=self.project_name, **kwargs)
        return results

    def validate(self):
        metrics = self.model.val()
        return metrics

    def export(self, format="onnx"):
        path = self.model.export(format=format)
        return path
