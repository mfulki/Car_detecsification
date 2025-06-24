from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import EfficientNetB3

from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
# from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils
import time
import cv2
import os
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, CSVLogger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

args = {
    "dataset": "/mnt/d/belajar s2/PNU/Image Clas/Classification/dataset",
    "model": "/mnt/d/belajar s2/PNU/Image Clas/Classification/classification2.keras",
    "plot":"/mnt/d/belajar s2/PNU/Image Clas/Classification/plot2",
    "conv":"/mnt/d/belajar s2/PNU/Image Clas/Classification/conv2",
}

EPOCHS = 25
BS = 32

imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    print(f"Memproses: {imagePath}")
    try:
        label = imagePath.split(os.path.sep)[-2]
        print(f"Label: {label}")

        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        data.append(image)
        labels.append(label)
    except Exception as e:
        print(f"Error inside {imagePath}: {e}")

data = np.array(data, dtype="float32")
labels = np.array(labels)
print("labels : ",labels)

le = LabelEncoder()
labels = le.fit_transform(labels)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

labels = to_categorical(labels)
print(le.classes_)
car_class=len(le.classes_)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
print("Distribusi training set:", np.bincount(trainY.argmax(axis=1)))
print("Distribusi testing set:", np.bincount(testY.argmax(axis=1)))

aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.25,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.25,
	horizontal_flip=True,
	fill_mode="nearest")

baseModel = EfficientNetB3(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))


headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(car_class, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False
 
initial_learning_rate = 1e-4
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=EPOCHS,
    decay_rate=0.96,
    staircase=True  
)
opt = Adam(learning_rate=lr_schedule)


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
csv_logger = CSVLogger('training_log.csv', append=True)

model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
    callbacks=[early_stop,csv_logger],
	validation_data=(testX, testY),
	epochs=EPOCHS,class_weight=class_weight_dict)

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=le.classes_))

print("Menyimpan model")
model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
N = len(H.history["loss"])
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args.get("plot", "plot2.png"))  
plt.show()  

cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig(args.get("conv", "conv2.png"))  
plt.show()
