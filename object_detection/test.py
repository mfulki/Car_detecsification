from ultralytics import YOLO
import cv2
import tensorflow as tf
import numpy as np
import time

from tensorflow.keras.applications.efficientnet_v2 import preprocess_input


class_names=['Box','Hatchback','MPV','Other','Pick Up','Sedan','Sport','SUV','Van']


detector = YOLO("runs/train/vehicle/weights/best.pt")
classifier = tf.keras.models.load_model("classification2.keras")


cap = cv2.VideoCapture("traffic_test.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka video.")
        break

    print("Frame terbaca, deteksi...")
    start = time.time()
    results = detector(frame,conf=0.5)[0]

    for r in results:
        for box in r.boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            dlabel = detector.names[cls_id]
            if dlabel=="car":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                
                
                resized = cv2.resize(crop, (224, 224))
                resized = resized.astype(np.float32)
                input_tensor=np.expand_dims(resized, axis=0)
                
                pred = classifier.predict(input_tensor)
                if cv2.waitKey(100) == 27:
                    break
                print("pred",pred)
                class_id = tf.argmax(pred[0]).numpy()

                label = class_names[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {dlabel} {cls_id:.2f} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    end = time.time()  
    fps = 1 / (end - start)
    print(f"FPS: {fps:.2f}")

    
    cv2.imshow("Result", frame)
    if cv2.waitKey(33) == 27:
        break
    


cap.release()

cv2.destroyAllWindows()
