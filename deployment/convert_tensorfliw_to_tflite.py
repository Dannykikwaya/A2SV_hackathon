import tensorflow as tf

# Charge ton modèle
model = tf.keras.models.load_model('chemin_vers_ton_modele.h5')

# Convertis le modèle en TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Enregistre le modèle converti
with open('chemin_vers_ton_modele_tflite.tflite', 'wb') as f:
    f.write(tflite_model)
