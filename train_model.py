# ===========================
# FishVision - Final Training Script
# ===========================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, callbacks
import os

# ===========================
# PATH CONFIGURATION
# ===========================
TRAIN_DIR = r"D:\Fish Img Classification\Dataset\train"
VAL_DIR = r"D:\Fish Img Classification\Dataset\val"
TEST_DIR = r"D:\Fish Img Classification\Dataset\test"

# ===========================
# IMAGE GENERATORS
# ===========================
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# ===========================
# CLASS INFORMATION
# ===========================
num_classes = len(train_data.class_indices)
print("\n✅ Classes found:", train_data.class_indices)

# Save class labels for use in Streamlit app
labels = [k for k, v in sorted(train_data.class_indices.items(), key=lambda x: x[1])]
with open("labels.txt", "w") as f:
    f.write("\n".join(labels))

# ===========================
# MODEL ARCHITECTURE
# ===========================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze base layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===========================
# CALLBACKS
# ===========================
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint('fishvision_model.h5', monitor='val_accuracy', save_best_only=True)

# ===========================
# TRAIN MODEL
# ===========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=[early_stop, checkpoint]
)

print("\n🎯 Training complete! Model saved as 'fishvision_model.h5'")
