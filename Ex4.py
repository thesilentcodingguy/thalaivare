import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.models import Model

PATH = r"D:\dataset"
X, captions, paths, labels = [], [], [], []

for label in sorted(os.listdir(PATH)):
    folder = os.path.join(PATH, label)

    if not os.path.isdir(folder):
        continue

    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        try:
            im = Image.open(os.path.join(folder, f)).convert("RGB")
            X.append(np.array(im.resize((224, 224))))
            captions.append(
                f"<start> a photo of a {label.replace('_', ' ').lower()} <end>"
            )
            paths.append(os.path.join(folder, f))
            labels.append(label)
        except Exception as e:
            print("Skipped:", f, e)

if len(X) < 5:
    raise ValueError("Dataset must contain at least 5 valid images.")

X = np.array(X, dtype=np.float32)
print("Total images:", len(X))

cnn = VGG16(
    weights="imagenet",
    include_top=False,
    pooling="max",
    input_shape=(224, 224, 3)
)

cnn.trainable = False

features = cnn.predict(
    preprocess_input(X.copy()),
    batch_size=16,
    verbose=1
)

print("Feature shape:", features.shape)

tok = Tokenizer(filters="", lower=True)
tok.fit_on_texts(captions)
seq = tok.texts_to_sequences(captions)
V = len(tok.word_index) + 1
L = max(map(len, seq))

print("Vocabulary size:", V)
print("Maximum caption length:", L)

# 4. Split images before creating training sequences
indices = np.arange(len(X))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

def make_data(indices):
    a, b, y = [], [], []
    for i in indices:
        s = seq[i]
        for j in range(1, len(s)):
            a.append(features[i])
            b.append(s[:j])
            y.append(s[j])

    b = pad_sequences(
        b, maxlen=L - 1, padding="post"
    )

    return np.array(a), b, np.array(y)

a_train, b_train, y_train = make_data(train_idx)
a_test, b_test, y_test = make_data(test_idx)

print("Training examples:", len(y_train))
print("Testing examples:", len(y_test))

image = Input(shape=(features.shape[1],))
text = Input(shape=(L - 1,))

h = Dense(256, activation="tanh")(image)
c = Dense(256, activation="tanh")(image)
e = Embedding(V, 128, mask_zero=True)(text)
z = LSTM(256)(e, initial_state=[h, c])

out = Dense(V, activation="softmax")(z)

model = Model([image, text], out)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    [a_train, b_train],
    y_train,
    validation_data=([a_test, b_test], y_test),
    epochs=30,
    batch_size=32
)

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN-LSTM Training Accuracy")
plt.legend()
plt.show()

def caption(path):
    im = Image.open(path).convert("RGB").resize((224, 224))
    im = np.array(im, dtype=np.float32)[None, ...]
    im = preprocess_input(im)

    f = cnn.predict(im, verbose=0)
    words = ["<start>"]

    for _ in range(L - 1):
        s = tok.texts_to_sequences([" ".join(words)])
        s = pad_sequences(s, maxlen=L - 1, padding="post")
        p = model.predict([f, s], verbose=0)[0]
        w = tok.index_word.get(int(np.argmax(p)), "")
        if w == "<end>" or not w:
            break
        words.append(w)

    return " ".join(words[1:])

for i in test_idx[:5]:
    expected = captions[i]
    generated = caption(paths[i])

    print("\nImage:", paths[i])
    print("Expected:", expected)
    print("Generated:", generated)
    plt.figure(figsize=(5, 4))
    plt.imshow(Image.open(paths[i]).convert("RGB"))
    plt.title("Generated: " + generated)
    plt.axis("off")
    plt.show()

model.save("image_caption_model.keras")
print("Model saved successfully.")
