import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    decode_predictions,
    preprocess_input,
)

# ==========================================
# 1. LOAD IMAGENET SUBSET (IMAGENETTE)
# ==========================================
print("Loading Imagenette dataset from TFDS...")
dataset, info = tfds.load(
    "imagenette/320px", split="train", with_info=True, as_supervised=True
)
class_names = info.features["label"].names

# ==========================================
# 2. DATASET VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 4))
for i, (image, label) in enumerate(dataset.take(4)):
    plt.subplot(1, 4, i + 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title(f"Label: {class_names[label.numpy()]}")
    plt.axis("off")
plt.suptitle("Sample Images from Imagenette Subset", fontsize=14)
plt.tight_layout()
plt.show()

# ==========================================
# 3. LOAD PRETRAINED VGG16 MODEL
# ==========================================
vgg_model = VGG16(weights="imagenet", include_top=True)

# Grab a single sample image for prediction & visualization
for raw_image, raw_label in dataset.take(1):
    sample_img = raw_image.numpy()
    sample_label = raw_label.numpy()
    break

# Resize image to VGG16 input specs (224x224) and preprocess
img_resized = tf.image.resize(sample_img, (224, 224))
img_batch = tf.expand_dims(img_resized, axis=0)
img_preprocessed = preprocess_input(img_batch.numpy().copy())

# ==========================================
# 4. PREDICTION USING VGG16
# ==========================================
predictions = vgg_model.predict(img_preprocessed)
decoded_preds = decode_predictions(predictions, top=3)[0]

print("\n--- PREDICTION RESULTS ---")
print(f"True Dataset Class: {class_names[sample_label]}")
print("Top-3 VGG16 ImageNet Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    print(f"  {i+1}. {label}: {score * 100:.2f}%")

# Plot prediction alongside the original image
plt.figure(figsize=(5, 5))
plt.imshow(sample_img.astype("uint8"))
plt.title(
    f"True: {class_names[sample_label]}\nPred: {decoded_preds[0][1]} ({decoded_preds[0][2]*100:.1f}%)"
)
plt.axis("off")
plt.show()

# ==========================================
# 5. CNN FEATURE MAP VISUALIZATION
# ==========================================
# Target convolutional layers across different blocks
target_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
layer_outputs = [vgg_model.get_layer(name).output for name in target_layers]

# Build intermediate activation model
activation_model = tf.keras.Model(
    inputs=vgg_model.input, outputs=layer_outputs
)
activations = activation_model.predict(img_preprocessed)

# Visualize first 6 feature maps (filters) for each selected layer
num_filters_to_show = 6
fig, axes = plt.subplots(
    len(target_layers), num_filters_to_show, figsize=(15, 10)
)

for layer_idx, feature_map in enumerate(activations):
    for filter_idx in range(num_filters_to_show):
        ax = axes[layer_idx, filter_idx]
        # Extract feature map for filter_idx
        f_map = feature_map[0, :, :, filter_idx]

        ax.imshow(f_map, cmap="viridis")
        ax.axis("off")

        if filter_idx == 0:
            ax.set_title(
                target_layers[layer_idx],
                fontsize=11,
                fontweight="bold",
                loc="left",
            )

plt.suptitle(
    "CNN Feature Map Visualizations across VGG16 Layers", fontsize=15, y=0.95
)
plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.show()
