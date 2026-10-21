import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

SIZE, NOISE, EPOCHS = 64, 100, 20

# 1. Load images and captions
images, captions = [], []

for label in os.listdir("images"):
    folder = os.path.join("images", label)
    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):
        try:
            img = Image.open(os.path.join(folder, file))
            img = img.convert("RGB").resize((SIZE, SIZE))
            images.append(np.array(img))
            captions.append("a photo of " + label.replace("_", " "))
        except:
            pass

images = np.array(images, dtype="float32") / 127.5 - 1

if len(images) == 0:
    raise ValueError("No images found. Check the images folder.")

# 2. Encode text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
MAX_LEN = max(len(c.split()) for c in captions)

texts = pad_sequences(
    tokenizer.texts_to_sequences(captions),
    maxlen=MAX_LEN, padding="post"
)
VOCAB = len(tokenizer.word_index) + 1

# 3. Text encoder
text_input = Input(shape=(MAX_LEN,))
x = Embedding(VOCAB, 16)(text_input)
text_features = GlobalAveragePooling1D()(x)
encoder = Model(text_input, text_features)

# 4. Generator: noise + text -> image
noise_input = Input(shape=(NOISE,))
caption_input = Input(shape=(MAX_LEN,))
features = encoder(caption_input)

x = Concatenate()([noise_input, features])
x = Dense(8 * 8 * 128, activation="relu")(x)
x = Reshape((8, 8, 128))(x)
x = Conv2DTranspose(64, 4, strides=2, padding="same",
                    activation="relu")(x)
x = Conv2DTranspose(3, 4, strides=4, padding="same",
                    activation="tanh")(x)
generator = Model([noise_input, caption_input], x)

# 5. Discriminator: image + text -> real/fake
image_input = Input(shape=(SIZE, SIZE, 3))
disc_text = Input(shape=(MAX_LEN,))

features = encoder(disc_text)
features = Dense(SIZE * SIZE, activation="relu")(features)
features = Reshape((SIZE, SIZE, 1))(features)

x = Concatenate()([image_input, features])
x = Conv2D(32, 4, strides=2, padding="same")(x)
x = LeakyReLU(negative_slope=0.2)(x)
x = Flatten()(x)
output = Dense(1)(x)
discriminator = Model([image_input, disc_text], output)

# 6. Loss and optimizers
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_opt = tf.keras.optimizers.Adam(0.0002)
d_opt = tf.keras.optimizers.Adam(0.0002)

# 7. Train GAN
dataset = tf.data.Dataset.from_tensor_slices((images, texts))
dataset = dataset.shuffle(len(images)).batch(16)

for epoch in range(EPOCHS):
    g_losses, d_losses = [], []

    for real_img, real_text in dataset:
        batch_size = tf.shape(real_img)[0]
        noise = tf.random.normal((batch_size, NOISE))

        with tf.GradientTape() as tape:
            fake_img = generator([noise, real_text], training=True)
            real_out = discriminator([real_img, real_text], training=True)
            fake_out = discriminator([fake_img, real_text], training=True)

            d_loss = (
                loss_fn(tf.ones_like(real_out), real_out) +
                loss_fn(tf.zeros_like(fake_out), fake_out)
            )

        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_opt.apply_gradients(zip(grads, discriminator.trainable_variables))

        noise = tf.random.normal((batch_size, NOISE))

        with tf.GradientTape() as tape:
            fake_img = generator([noise, real_text], training=True)
            fake_out = discriminator([fake_img, real_text], training=False)
            g_loss = loss_fn(tf.ones_like(fake_out), fake_out)

        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_opt.apply_gradients(zip(grads, generator.trainable_variables))

        g_losses.append(float(g_loss))
        d_losses.append(float(d_loss))

    print(f"Epoch {epoch + 1}/{EPOCHS} | "
          f"G Loss: {np.mean(g_losses):.4f} | "
          f"D Loss: {np.mean(d_losses):.4f}")

# 8. Generate an image from text
def generate_image(description):
    text = pad_sequences(
        tokenizer.texts_to_sequences([description]),
        maxlen=MAX_LEN, padding="post"
    )

    noise = tf.random.normal((1, NOISE))
    result = generator.predict([noise, text], verbose=0)[0]

    plt.imshow(np.clip((result + 1) / 2, 0, 1))
    plt.title(description)
    plt.axis("off")
    plt.show()

# 9. Generate and save
generate_image("a photo of tiger")
generator.save("text_gan.keras")
print("Generator saved successfully.")
