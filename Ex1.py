import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.utils import load_img, img_to_array


data_path = "/content/drive/MyDrive/images"

classes = [
    "Horses",
    "Foods",
    "Bus",
    "Mountains",
    "Dinosaur",
    "Monuments",
    "TribalPeople",
    "Elephant",
    "Beaches",
    "Flowers"
]

print("Number of classes:", len(classes))
print("Classes:", classes)

vgg = VGG16(
    weights="imagenet",
    include_top=True
)

print("\nVGG16 Input Shape :", vgg.input_shape)
print("VGG16 Output Shape:", vgg.output_shape)


selected_class = random.choice(classes)

folder = os.path.join(
    data_path,
    selected_class
)

image_name = random.choice(
    os.listdir(folder)
)

image_path = os.path.join(
    folder,
    image_name
)

print("\nOriginal Class:", selected_class)
print("Image:", image_name)

img = load_img(
    image_path,
    target_size=(224, 224)
)

print("Image size:", img.size)


plt.figure(figsize=(5, 5))

plt.imshow(img)
plt.title(f"Original Class: {selected_class}")
plt.axis("off")

plt.show()



x = img_to_array(img)

print("Image array shape:", x.shape)


x = np.expand_dims(x, axis=0)

print("Input shape after batch dimension:", x.shape)

x = preprocess_input(x)

print("Preprocessing completed.")

predictions = vgg.predict(x)

print("\nPrediction shape:", predictions.shape)

top5 = decode_predictions(
    predictions,
    top=5
)[0]

print("\nTop-5 ImageNet Predictions:")
print("--------------------------------")

for rank, (_, name, probability) in enumerate(top5, start=1):

    print(
        f"{rank}. {name:25s} : {probability:.2%}"
    )

plt.figure(figsize=(7, 5))
plt.imshow(img)
plt.title(f"Original Class: {selected_class}")
plt.axis("off")
plt.show()

print("Top-5 ImageNet Predictions:\n")

for rank, (_, name, probability) in enumerate(top5, start=1):

    print(
        f"{rank}. {name:25s} : {probability:.2%}"
    )


print("\nVGG16 Layers:")
print("--------------------------------")

for i, layer in enumerate(vgg.layers):

    print(
        i,
        layer.name,
        layer.output.shape
    )

layers_to_extract = [
    "block1_conv2",
    "block3_conv3",
    "block5_conv3"
]

print("\nSelected Layers:")
print(layers_to_extract)

feature_models = {}

for layer_name in layers_to_extract:

    feature_models[layer_name] = tf.keras.Model(
        inputs=vgg.input,
        outputs=vgg.get_layer(layer_name).output
    )

features = {}

for layer_name, feature_model in feature_models.items():

    features[layer_name] = feature_model.predict(x)

    print(
        f"{layer_name}: {features[layer_name].shape}"
    )

for layer_name, feature in features.items():

    plt.figure(figsize=(10, 5))

    for i in range(8):

        plt.subplot(2, 4, i + 1)

        plt.imshow(
            feature[0, :, :, i],
            cmap="gray"
        )

        plt.axis("off")

    plt.suptitle(
        f"Feature Maps - {layer_name}"
    )

    plt.show()


print("\nIntermediate Feature Map Dimensions:")
print("------------------------------------------")

for layer_name, feature in features.items():

    print(
        f"{layer_name:15s} -> {feature.shape}"
    )

filter_layer = vgg.get_layer(
    "block1_conv1"
)

filters, biases = filter_layer.get_weights()

print("\nFirst Convolutional Layer Filters:")
print("Filter shape:", filters.shape)

plt.figure(figsize=(10, 6))

for i in range(16):

    f = filters[:, :, :, i]
    f = (
        f - f.min()
    ) / (
        f.max() - f.min()
    )

    plt.subplot(4, 4, i + 1)

    plt.imshow(f)

    plt.axis("off")

plt.suptitle(
    "First 16 Learned Filters - VGG16 block1_conv1"
)

plt.show()

print("Input image          :", (224, 224, 3))
print("Batch input          :", x.shape)
print("VGG16 output         :", predictions.shape)
print("Top-5 predictions    :", len(top5))

print("\nIntermediate layers:")

for layer_name, feature in features.items():

    print(
        f"{layer_name:15s} -> {feature.shape}"
    )

print("\nExperiment completed successfully.")
