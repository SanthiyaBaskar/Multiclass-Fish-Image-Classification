import tensorflow as tf
import os

# Check if .h5 model exists
if not os.path.exists("fishvision_model.h5"):
    print("❌ fishvision_model.h5 not found in this folder!")
else:
    print("✅ Found fishvision_model.h5, converting to TensorFlow Lite...")

    # Load and convert model
    model = tf.keras.models.load_model("fishvision_model.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save converted model
    output_path = "fishvision_model.tflite"
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"✅ Conversion successful! Saved as {output_path}")
