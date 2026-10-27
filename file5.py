import os
import sys

# 1. Force the operating system level to completely skip SSL verification for URLs
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers, callbacks

DATASET_PATH = 'dataset/'

# 2. Fix class counting by ignoring hidden Mac system files like .DS_Store
classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')]
num_classes = len(classes)
class_mode = "binary" if num_classes == 2 else "categorical"

print(f" Detected Classes: {classes} (Total: {num_classes})")
print(f" Mode Chosen: {class_mode}")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# 3. Modern & Fast Data Loading
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=class_mode
)

val_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=class_mode
)

# 4. Optimise Data Pipeline Performance
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)

# 5. Define Robust Augmentation
data_augmentation = Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

rescaling = layers.Rescaling(scale=1./127.5, offset=-1)

# 6. Build the Pre-Trained Architecture
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    layers.Input(shape=(128, 128, 3)),
    data_augmentation,
    rescaling,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid") if class_mode == "binary" else layers.Dense(num_classes, activation="softmax")
])

# 7. Compile Configuration
loss_function = "binary_crossentropy" if class_mode == "binary" else "categorical_crossentropy"

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss=loss_function,
    metrics=["accuracy"]
)

# 8. Early Stopping to Avoid Overfitting
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 9. Train, Evaluate, and Save
model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[early_stopping]
)

test_loss, test_accuracy = model.evaluate(val_data)
print(f"\nFinal Model accuracy on validation data: {test_accuracy:.2f}")

model.save("image_classifier.keras")
