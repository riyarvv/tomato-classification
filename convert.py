import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('tomatofinal.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# This ensures it uses standard, compatible operations
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

# Save the new file
with open('tomato_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("New tomato_model.tflite created successfully!")