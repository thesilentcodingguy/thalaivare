import os
from PIL import Image
from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions
)
import numpy as np

DATASET = "dataset"
model = VGG16(weights="imagenet")

folders = sorted(os.listdir(DATASET))[:10]

for folder in folders:
    path = os.path.join(DATASET, folder)

    if not os.path.isdir(path):
        continue

    print(f"\nFolder {folder}:")

    for file in os.listdir(path)[:3]:
        try:
            img = Image.open(os.path.join(path, file)).convert("RGB")
            img = img.resize((224, 224))

            x = np.expand_dims(np.array(img), axis=0)
            x = preprocess_input(x)

            prediction = model.predict(x, verbose=0)
            label = decode_predictions(prediction, top=1)[0][0][1]

            print(f"  {file} -> {label}")

        except Exception:
            pass
