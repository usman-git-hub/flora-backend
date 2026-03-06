import tensorflow as tf

print("Loading original model...")
model = tf.keras.models.load_model('my_optimized_flower_model.keras')

print("Converting to Legacy TFLite format for version 2.14...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# This is the "Magic Fix": Force the use of older op versions
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.target_spec.supported_types = [tf.float32]

tflite_model = converter.convert()

with open('flower_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Success! Legacy 'flower_model.tflite' created.")