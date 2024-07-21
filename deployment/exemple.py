import tensorflow as tf
import numpy as np
import cv2

# Charge le modèle TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path='chemin_vers_ton_modele_tflite.tflite')
interpreter.allocate_tensors()

# Obtenir les détails des tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ouvrir la caméra
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prétraitement de l'image
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    img = img / 255.0

    # Faire une prédiction
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output_data[0])

    # Afficher la prédiction sur l'image
    cv2.putText(frame, f'Prediction: {pred}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
