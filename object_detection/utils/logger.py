import os
import sys
import logging
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="logs", prefix="train"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

        sys.stdout = open(self.log_path, 'w')
        sys.stderr = sys.stdout

        logging.basicConfig(level=logging.INFO)
        logging.info(f"Logging started at {timestamp}")
        logging.info(f"Log file: {self.log_path}")

    def log(self, message: str):
        logging.info(message)

    def close(self):
        sys.stdout.close()
