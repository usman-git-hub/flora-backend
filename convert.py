import tensorflow as tf

# Load your existing model
model = tf.keras.models.load_model('my_optimized_flower_model.keras')

# Convert to TFLite with basic optimization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # This shrinks the size further
tflite_model = converter.convert()

# Save the new version
with open('flower_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Success! 'flower_model.tflite' created.")