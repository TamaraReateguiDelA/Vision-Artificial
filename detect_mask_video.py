
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

#creamos la funcion de deteccion de rostros con mascarillas
def detect_and_predict_mask(frame, faceNet, maskNet):
	# Toma las dimensiones del frame y contruye un large binary object de el.

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pasa el blob por la red y se obtiene la deteccion de rostros
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# inicializar la lista de rostros y sus correspondientes ubicaciones,
	# y la lista de predicciones de mascarillas.
	faces = []
	locs = []
	preds = []

	# loop sobre las detecciones
	for i in range(0, detections.shape[2]):
		# extraer la confidencia (i.e., probabilidad) asociada con la detección
		confidence = detections[0, 0, i, 2]

		# filtrar las detecciones debiles asegurando que la confidencia es mayor a la confidencia minima
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Asegurar que los BB no saldran del frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extraer la ROI del rostro, convertirlo de BGR a RGB
			# ordenar, reajustar medidas a 224x224, procesarlas
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# añadir los rostros y sus respectivos bounding boxes a las listas
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# solo hacer predicciones si al menos existe un rostro detectado
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# retornar una tupla de las predicciones de rostros con mascarillas y sus respectivos locaciones.
	return (locs, preds)

# Cargamos el face detector serializado
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Cargamos el mask detector creado anteriormente
maskNet = load_model("mask_detector.model")

# Inicializamos el video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop sobre los frames del  video stream
while True:
	# El frame es reajustado para tener un maximo with de 400 pixeles
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# Se detectan rostros en el frame (face detector) y se predice si usan o no mascarillas (mask detector)
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop sobre las locaciones de los rostros, si posee mascarilla se dibuja un bounding box color verde, si no posee mascarilla
	# un bounding box rojo
	for (box, pred) in zip(locs, preds):
		# desempaquetar bounding box y predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determinar la clase de la etiqueta y el color que se usará para dibujar el BB y el texto

		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# incluir la probailidad en la etiqueta
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# mostrar la etiqueta y el bounding box en la salida del frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# mostrar el output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Si la tecla `q` key ha sido presionada, se sale del loop
	if key == ord("q"):
		break

#limpiar
cv2.destroyAllWindows()
vs.stop()